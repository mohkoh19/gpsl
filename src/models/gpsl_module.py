"""
GPSLModule: PyTorch Lightning module implementing the Global Parallel Split Learning (GPSL) training loop.
Handles manual optimization, model splitting, distributed client simulation, and logging.
"""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import AUROC, Accuracy, F1Score, MeanMetric

from utils.aggregation import grad_avg, replace_bn_with_gn
from utils.splitter import split_model


class GPSLModule(LightningModule):
    """
    PyTorch Lightning module implementing GPSL.

    This module splits a neural network into client-side and server-side parts,
    simulates parallel training on multiple clients, and applies server-side optimization.
    """

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
        """
        Initialize the GPSLModule.

        Parameters
        ----------
        model : Callable
            A callable that returns a new instance of the full model.
        optimizer : Callable
            A callable that returns a new optimizer instance.
        loss_fn : Callable
            Loss function used for training.
        cut_layer : str
            Name of the layer to split the model at.
        gn_num_groups : int
            Number of groups for GroupNorm.
        num_clients : int
            Number of simulated clients.
        max_workers : int
            Maximum number of threads used for client forward passes.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="gpsl_client_forward"
        )
        self.iters = None

    def setup(self, stage):
        """
        Called at the beginning of training to prepare models and metrics.

        Parameters
        ----------
        stage : str
            Current training stage ('fit' or 'test').
        """
        if stage != "fit":
            return

        num_classes = self.trainer.datamodule.num_classes
        self.model = self.hparams.model(num_classes).to(self.device)
        client_model, server_model = split_model(self.model, self.hparams.cut_layer)
        client_model = replace_bn_with_gn(client_model, self.hparams.gn_num_groups)

        self.client_models = [
            deepcopy(client_model).to(self.device)
            for _ in range(self.hparams.num_clients)
        ]
        for cm in self.client_models:
            cm.train()

        self.server_model = server_model.to(self.device)

        self.train_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.train_acc = Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_acc = Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.train_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.train_auroc = AUROC(task="multiclass", num_classes=num_classes)
        self.test_auroc = AUROC(task="multiclass", num_classes=num_classes)

    def on_train_epoch_start(self):
        """
        Prepare new iterators for each client's dataloader at the start of the epoch.
        """
        self.iters = [
            iter(dl) for dl in self.trainer.datamodule.train_subset_dataloaders
        ]

    def on_train_epoch_end(self):
        """
        Reset training metrics at the end of the training epoch.
        """
        self.train_acc.reset()
        self.train_f1.reset()
        self.train_auroc.reset()

    def on_validation_epoch_end(self):
        """
        Reset validation metrics at the end of validation epoch.
        """
        self.test_acc.reset()
        self.test_f1.reset()
        self.test_auroc.reset()
        self.client_models[-1].train()

    def forward(self, active_indices):
        """
        Perform a forward pass for a given set of active client indices.

        Parameters
        ----------
        active_indices : list of int
            Indices of clients participating in the current step.

        Returns
        -------
        preds : torch.Tensor
            Server model output.
        y : torch.Tensor
            Ground truth labels from all clients.
        """

        def client_forward(i):
            try:
                x, y = next(self.iters[i])
                self.client_optims[i].zero_grad()
                sd = self.client_models[i](x.to(self.device))
                return sd, y.to(self.device)
            except StopIteration:
                return None, None

        client_data = list(self.executor.map(client_forward, active_indices))
        B = [cd[0] for cd in client_data if cd[0] is not None]
        y = [cd[1] for cd in client_data if cd[0] is not None]
        sd = torch.concatenate(B, axis=0)
        y = torch.concatenate(y, axis=0)

        self.server_optimizer.zero_grad()
        preds = self.server_model(sd)
        return preds, y

    def training_step(self, active_indices, batch_idx):
        """
        One training step consisting of client forward passes, server-side loss and gradient updates.

        Parameters
        ----------
        active_indices : list of int
            Indices of the active clients for this batch.
        batch_idx : int
            Index of the current training batch.

        Returns
        -------
        torch.Tensor
            Computed training loss.
        """
        preds, y = self.forward(active_indices)
        loss = self.hparams.loss_fn(preds, y)

        self.manual_backward(loss)
        self.client_models = grad_avg(self.client_models, active_indices)

        self.server_optimizer.step()
        for co in self.client_optims:
            co.step()

        self.log("train/loss", loss)
        self.log("train/acc", self.train_acc(preds, y))
        self.log("train/f1", self.train_f1(preds, y))
        self.log("train/auroc", self.train_auroc(preds, y))

        def batch_deviation(targets: torch.Tensor) -> torch.Tensor:
            """
            Compute the L1 deviation of the batch class distribution from the global distribution.

            Parameters
            ----------
            targets : torch.Tensor
                Batch target labels.

            Returns
            -------
            torch.Tensor
                L1 deviation value.
            """
            num_classes = self.trainer.datamodule.num_classes
            global_dist = self.trainer.datamodule.global_dist
            dist = torch.bincount(targets.detach().cpu(), minlength=num_classes).float()
            dist /= dist.sum()
            return F.l1_loss(dist, global_dist, reduction="sum")

        self.log("train/batch_deviation", batch_deviation(y))
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step using the last client model.

        Parameters
        ----------
        batch : tuple
            A batch containing (x, y) data from the validation set.
        batch_idx : int
            Index of the validation batch.

        Returns
        -------
        torch.Tensor
            Computed validation loss.
        """
        x, y = batch
        self.client_models[-1].eval()
        sd = self.client_models[-1](x)
        out = self.server_model(sd)
        loss = self.hparams.loss_fn(out, y)

        self.log("test/loss", loss)
        self.log("test/acc", self.test_acc(out, y))
        self.log("test/f1", self.test_f1(out, y))
        self.log("test/auroc", self.test_auroc(out, y))
        return loss

    def configure_optimizers(self):
        """
        Set up optimizers for both the server and all client models.

        Returns
        -------
        list
            List of optimizer instances.
        """
        self.server_optimizer = self.hparams.optimizer(self.server_model.parameters())
        self.client_optims = [
            self.hparams.optimizer(cm.parameters()) for cm in self.client_models
        ]
        return self.client_optims + [self.server_optimizer]
