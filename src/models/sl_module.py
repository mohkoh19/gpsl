from copy import deepcopy
import torch
from lightning import LightningModule
from utils.aggregation import (
    replace_bn_with_gn,
)
from utils.splitter import split_model
from torchmetrics import (
    AUROC,
    Accuracy,
    F1Score,
    MeanMetric,
)


class SLModule(LightningModule):
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

        self.current_model = 0

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

    def forward(self, x):
        smashed_data = self.client_models[self.current_model](x)
        preds = self.server_model(smashed_data)

        return preds

    def training_step(self, batch, batch_idx):
        x, Y, dl_idx = batch

        self.server_optimizer.zero_grad()
        self.client_optims[self.current_model].zero_grad()

        # Forward pass
        preds = self.forward(x)
        loss = self.hparams.loss_fn(preds, Y)

        # Backward pass
        self.manual_backward(loss)

        # Optimizer step
        self.server_optimizer.step()
        self.client_optims[self.current_model].step()

        # Update the current model index
        max_idx = torch.max(dl_idx).item()
        if max_idx > self.current_model:
            # Copy model weights to the new client model
            self.client_models[max_idx].load_state_dict(
                self.client_models[self.current_model].state_dict()
            )

            self.current_model = max_idx

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

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."

        # Sync all client model weights
        new_model = self.client_models[self.current_model].state_dict()
        for i in range(self.hparams.num_clients):
            self.client_models[i].load_state_dict(new_model)

        # Reset the current model index
        self.current_model = 0

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # For testing, use the last client: use the same unpacking as in forward.
        out = self.forward(x)
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
