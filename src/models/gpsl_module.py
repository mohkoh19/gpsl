from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import torch
from lightning import LightningModule
from utils.aggregation import (
    grad_avg,
    replace_bn_with_gn,
)
from utils.splitter import split_model
from torchmetrics import (
    AUROC,
    Accuracy,
    F1Score,
    MeanMetric,
)


class GPSLModule(LightningModule):
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        cut_layer,
        gn_num_groups,
        num_clients,
    ):
        super().__init__()
        # Save all hyperparameters so that they can be accessed via self.hparams
        self.save_hyperparameters(logger=False)

        # Disable automatic optimization (we are doing manual gradient handling)
        self.automatic_optimization = False

        # ThreadPoolExecutor will be initialized each epoch in on_train_epoch_start
        self.executor = ThreadPoolExecutor(
            max_workers=num_clients, thread_name_prefix="gpsl_client_model"
        )

    def setup(self, stage):
        if stage != "fit":
            return

        # Get num_classes from datamodule
        num_classes = self.trainer.datamodule.num_classes

        # Instantiate the model
        self.model = self.hparams.model(num_classes).to(self.device)

        # Split the model into client and server parts
        client_model, server_model = split_model(self.model, self.hparams.cut_layer)

        # Replace BatchNorm with GroupNorm in the client model
        client_model = replace_bn_with_gn(client_model, self.hparams.gn_num_groups)

        # Create one client model per client (each on its designated device)
        self.client_models = [
            deepcopy(client_model).to(self.device)
            for i in range(self.hparams.num_clients)
        ]

        # Assign the server model
        self.server_model = server_model.to(self.device)

        # Metrics
        self.train_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes)

        self.train_auroc = AUROC(task="multiclass", num_classes=num_classes)
        self.test_auroc = AUROC(task="multiclass", num_classes=num_classes)

    def forward(self, x, active_indices=None):
        # If active_indices is not provided, use all clients
        if active_indices is None:
            active_indices = list(range(len(x)))

        def client_forward(i):
            sd = self.client_models[i](x[i])
            return sd

        # Collect the smashed outputs and labels from participating clients
        smashed_data = list(self.executor.map(client_forward, active_indices))

        # Concatenate smashed data along the batch dimension
        smashed_data = torch.concatenate(smashed_data, axis=0)

        preds = self.server_model(smashed_data)
        return preds

    def training_step(self, batch, batch_idx):
        # The incoming batch is from a CombinedDataLoader
        # It is a list of batches, one for each client
        active_indices = [i for i, b in enumerate(batch) if b is not None]

        x = [b[0] if b is not None else None for b in batch]
        Y = torch.concatenate([batch[i][1] for i in active_indices], axis=0)

        # Server forward pass and loss computation
        self.server_optimizer.zero_grad()
        for co in self.client_optims:
            co.zero_grad()

        preds = self.forward(x, active_indices)
        loss = self.hparams.loss_fn(preds, Y)

        # Perform gradient averaging over active clients + Optimizer step
        self.manual_backward(loss)

        # Perform gradient averaging over active clients
        self.client_models = grad_avg(self.client_models, active_indices)

        # Optimizer steps
        self.server_optimizer.step()
        for co in self.client_optims:
            co.step()

        # Logging
        self.log(
            "train/loss",
            loss,
        )
        self.log(
            "train/acc",
            self.train_acc(preds, Y),
        )
        self.log(
            "train/f1",
            self.train_f1(preds, Y),
        )
        self.log(
            "train/auroc",
            self.train_auroc(preds, Y),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # For testing, use the last client: use the same unpacking as in forward.
        sd = self.client_models[-1](x)
        out = self.server_model(sd)
        loss = self.hparams.loss_fn(out, y)

        # Logging
        self.log(
            "test/loss",
            loss,
        )
        self.log(
            "test/acc",
            self.test_acc(out, y),
        )
        self.log(
            "test/f1",
            self.test_f1(out, y),
        )
        self.log(
            "test/auroc",
            self.test_auroc(out, y),
        )

        return loss

    def configure_optimizers(self):
        # Create an optimizer for the server model and one for each client model
        self.server_optimizer = self.hparams.optimizer(self.server_model.parameters())
        self.client_optims = [
            self.hparams.optimizer(cm.parameters()) for cm in self.client_models
        ]
        return self.client_optims + [self.server_optimizer]
