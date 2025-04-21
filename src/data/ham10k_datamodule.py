from ast import List
from typing import Any, Dict, Optional
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from lightning import LightningDataModule

from src.data.components.ham10k import HAM10000


class HAM10KDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        transforms: List = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 7

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose(self.hparams.transforms.train)
        test_transforms = transforms.Compose(self.hparams.transforms.test)

        self.data_train = HAM10000(
            root=self.hparams.data_dir, train=True, transform=train_transforms
        )
        self.data_test = HAM10000(
            root=self.hparams.data_dir, train=False, transform=test_transforms
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


if __name__ == "__main__":
    _ = HAM10KDataModule()
