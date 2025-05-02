"""
DistributedDataModule for GPSL: A wrapper around a base LightningDataModule to simulate
distributed training across multiple clients with custom sampling strategies.
"""

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset

from src.utils.sampling import GlobalBatchSampler, LocalBatchSampler
from src.utils.splitter import split_dataset


class DistributedDataModule(LightningDataModule):
    """
    Wraps a base LightningDataModule to simulate distributed client training
    by splitting the training dataset and scheduling batches using GPSL samplers.
    """

    def __init__(
        self,
        base_datamodule: LightningDataModule,
        num_clients: int,
        subset_config: dict,
        sampler: object,
    ):
        """
        Parameters
        ----------
        base_datamodule : LightningDataModule
            The base datamodule to wrap.
        num_clients : int
            Number of simulated clients.
        subset_config : dict
            Dictionary containing the distribution type and its parameters.
        sampler : object
            A sampling strategy class with a `generate_batches` method.
        """
        super().__init__()
        self.base = base_datamodule
        self.num_clients = num_clients
        self.num_classes = base_datamodule.num_classes
        self.subset_config = subset_config
        self.train_subset_dataloaders = None
        self.sampler = sampler

    def prepare_data(self):
        """Delegate data preparation to the base datamodule."""
        return self.base.prepare_data()

    def extract_loader_config(self, loader: DataLoader) -> dict:
        """
        Extracts reusable configuration from a DataLoader.

        Parameters
        ----------
        loader : DataLoader
            The source DataLoader.

        Returns
        -------
        dict
            Configuration dictionary (e.g., num_workers, pin_memory).
        """
        return {
            "num_workers": loader.num_workers,
            "pin_memory": loader.pin_memory,
            "timeout": loader.timeout,
            "worker_init_fn": loader.worker_init_fn,
            "persistent_workers": loader.persistent_workers,
        }

    def setup(self, stage=None):
        """
        Sets up the distributed training data by splitting the base dataset
        and assigning sampling logic to simulate client-based batches.

        Parameters
        ----------
        stage : str, optional
            The current stage ('fit', 'validate', etc.).
        """
        self.base.setup(stage)
        train_dataset = self.base.train_dataloader().dataset

        targets = (
            [target for _, target in train_dataset]
            if hasattr(train_dataset, "__getitem__")
            else None
        )

        self.global_dist = torch.bincount(
            torch.tensor(targets), minlength=self.num_classes
        ).float()
        self.global_dist /= self.global_dist.sum()

        split_indices = split_dataset(
            targets,
            self.num_clients,
            self.subset_config["distribution"],
            **self.subset_config["params"],
        )
        subsets = [Subset(train_dataset, indices) for indices in split_indices]

        base_loader = self.base.train_dataloader()
        loader_cfg = self.extract_loader_config(base_loader)

        self.train_subset_dataloaders = [
            DataLoader(
                dataset=subset, **loader_cfg, batch_sampler=LocalBatchSampler(subset)
            )
            for subset in subsets
        ]

        local_batch_samplers = [
            dl.batch_sampler for dl in self.train_subset_dataloaders
        ]

        self.global_batch_sampler = GlobalBatchSampler(
            self.sampler,
            local_batch_samplers,
            base_loader.batch_size,
        )

    def train_dataloader(self):
        """
        Returns a DataLoader that yields client indices per batch step,
        simulating a global schedule over distributed clients.

        Returns
        -------
        DataLoader
            DataLoader for the GPSL global batch sampler.
        """
        if self.train_subset_dataloaders is None:
            raise RuntimeError("setup() must be called before train_dataloader()")

        return DataLoader(self.global_batch_sampler, batch_size=1, shuffle=False)

    def on_train_epoch_start(self):
        """Refresh batch scheduling at the start of each training epoch."""
        self.global_batch_sampler.generate_batches()

    def val_dataloader(self):
        """Returns the validation dataloader from the base module."""
        return self.base.val_dataloader()

    def teardown(self, stage=None):
        """Delegate teardown to the base datamodule."""
        return self.base.teardown(stage)

    def state_dict(self):
        """Delegates state_dict collection to the base datamodule."""
        return self.base.state_dict()

    def load_state_dict(self, state_dict):
        """
        Loads the datamodule state.

        Parameters
        ----------
        state_dict : dict
            State dictionary to load.
        """
        return self.base.load_state_dict(state_dict)
