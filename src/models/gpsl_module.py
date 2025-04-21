from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import torch
from lightning import LightningModule
from utils.aggregation import (
    grad_avg,
    replace_bn_with_gn,
)
from utils.splitter import split_model
import gc, tracemalloc
from torchmetrics import (
    AUROC,
    Accuracy,
    F1Score,
    MeanMetric,
)
import multiprocessing as mp
import torch.nn.functional as F


class GPSLModule(LightningModule):
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        cut_layer,
        gn_num_groups,
        num_clients,
        max_workers=4,
    ):
        super().__init__()
        # Save all hyperparameters so that they can be accessed via self.hparams
        self.save_hyperparameters(logger=False)

        # Disable automatic optimization (we are doing manual gradient handling)
        self.automatic_optimization = False

        # ThreadPoolExecutor for client model forward passes
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="gpsl_client_forward"
        )

        # Reference to subset iterators
        self.iters = None

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

        for cm in self.client_models:
            cm.train()

        # Assign the server model
        self.server_model = server_model.to(self.device)

        # Metrics
        self.train_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        self.train_auroc = AUROC(task="multiclass", num_classes=num_classes)
        self.test_auroc = AUROC(task="multiclass", num_classes=num_classes)

    def on_train_epoch_start(self):
        # Get the iterators from the datamodule
        self.iters = [
            iter(dl) for dl in self.trainer.datamodule.train_subset_dataloaders
        ]

    def on_train_epoch_end(self):
        self.train_acc.reset()
        self.train_f1.reset()
        self.train_auroc.reset()

    def on_validation_epoch_end(self):
        self.test_acc.reset()
        self.test_f1.reset()
        self.test_auroc.reset()

        self.client_models[-1].train()

    def forward(self, active_indices):
        def client_forward(i):
            try:
                x, y = next(self.iters[i])
                self.client_optims[i].zero_grad()
                sd = self.client_models[i](x.to(self.device))
                return sd, y.to(self.device)
            except StopIteration:
                return None, None

        # Collect the smashed outputs and labels from participating clients
        client_data = list(self.executor.map(client_forward, active_indices))

        B = [cd[0] for cd in client_data if cd[0] is not None]
        y = [cd[1] for cd in client_data if cd[0] is not None]
        sd = torch.concatenate(B, axis=0)
        y = torch.concatenate(y, axis=0)

        self.server_optimizer.zero_grad()
        preds = self.server_model(sd)
        return preds, y

    def training_step(self, active_indices, batch_idx):
        preds, y = self.forward(active_indices)
        loss = self.hparams.loss_fn(preds, y)

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
            self.train_acc(preds, y),
        )
        self.log(
            "train/f1",
            self.train_f1(preds, y),
        )
        self.log(
            "train/auroc",
            self.train_auroc(preds, y),
        )

        def batch_deviation(targets: torch.Tensor) -> torch.Tensor:
            num_classes = self.trainer.datamodule.num_classes
            global_dist = self.trainer.datamodule.global_dist

            dist = torch.bincount(targets.detach().cpu(), minlength=num_classes).float()
            dist /= dist.sum()

            deviation = F.l1_loss(
                dist,
                global_dist,
                reduction="sum",
            )

            return deviation

        self.log(
            "train/batch_deviation",
            batch_deviation(y),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.client_models[-1].eval()
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
