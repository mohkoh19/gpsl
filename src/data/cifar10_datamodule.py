"""
LightningDataModule for CIFAR-10 dataset with configurable data loading and transformation.
Provides training and validation DataLoaders for use with PyTorch Lightning workflows.
"""

from typing import Any, Dict, Optional, List

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms


class CIFAR10DataModule(LightningDataModule):
    """
    PyTorch LightningDataModule for the CIFAR-10 dataset.

    This module handles data preparation, setup, and DataLoader creation for
    training and validation using user-defined transforms and loading options.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        transforms: List = None,
    ) -> None:
        """
        Initialize the CIFAR10DataModule.

        Parameters
        ----------
        data_dir : str
            Directory to download and store CIFAR-10 data.
        batch_size : int
            Number of samples per batch.
        num_workers : int
            Number of worker processes for data loading.
        pin_memory : bool
            Whether to pin memory in data loaders.
        persistent_workers : bool
            Whether to persist worker processes between epochs.
        transforms : list
            Dictionary containing 'train' and 'test' transformation pipelines.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        """
        Return the number of classes in the CIFAR-10 dataset.
        """
        return 10

    def prepare_data(self) -> None:
        """
        Download CIFAR-10 dataset if it is not already present.
        """
        CIFAR10(self.hparams.data_dir, train=True, download=True)
        CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up training and validation datasets with transforms.

        Parameters
        ----------
        stage : str, optional
            Stage identifier (e.g., 'fit', 'validate'), not used here.
        """
        train_transforms = transforms.Compose(self.hparams.transforms.train)
        test_transforms = transforms.Compose(self.hparams.transforms.test)

        self.data_train = CIFAR10(
            self.hparams.data_dir, train=True, transform=train_transforms
        )
        self.data_test = CIFAR10(
            self.hparams.data_dir, train=False, transform=test_transforms
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """
        Return the training DataLoader.

        Returns
        -------
        DataLoader
            DataLoader for the training dataset.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """
        Return the validation DataLoader.

        Returns
        -------
        DataLoader
            DataLoader for the validation dataset.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Optional cleanup after training/validation.

        Parameters
        ----------
        stage : str, optional
            Stage identifier.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """
        Return the state of the datamodule.

        Returns
        -------
        dict
            Empty dictionary (not used here).
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the state into the datamodule.

        Parameters
        ----------
        state_dict : dict
            State dictionary to load.
        """
        pass
