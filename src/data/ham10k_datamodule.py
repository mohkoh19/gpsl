from ast import List
from typing import Any, Dict, Optional
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from lightning import LightningDataModule
import os
from glob import glob

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class HAM10000(Dataset):
    def __init__(self, root, train=True, transform=None):
        """
        Args:
            root (string): Root directory of the dataset where the dataset csv file and images are located.
            train (bool): If True, loads the training split, else loads the test split.
            transform (callable, optional): Optional transform to be applied on a sample.
            random_state (int): Random seed for deterministic split.
        """
        self.root = os.path.join(root, "HAM10000")
        self.transform = transform

        self.df = pd.read_csv(os.path.join(self.root, "HAM10000_metadata.csv"))

        # Train-Test split
        train_df, test_df = train_test_split(self.df, test_size=0.2)
        self.df = train_df if train else test_df
        self.df.reset_index(inplace=True, drop=True)

        lesion_type = {
            "nv": "Melanocytic nevi",
            "mel": "Melanoma",
            "bkl": "Benign keratosis-like lesions ",
            "bcc": "Basal cell carcinoma",
            "akiec": "Actinic keratoses",
            "vasc": "Vascular lesions",
            "df": "Dermatofibroma",
        }

        imageid_path = {
            os.path.splitext(os.path.basename(x))[0]: x
            for x in glob(os.path.join(self.root, "*", "*.jpg"))
        }
        self.df["path"] = self.df["image_id"].map(imageid_path.get)
        self.df["cell_type"] = self.df["dx"].map(lesion_type.get)
        self.targets = pd.Categorical(self.df["cell_type"]).codes.astype(int).tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = Image.open(self.df["path"].iloc[idx])
        y = torch.tensor(self.targets[idx])

        if self.transform:
            x = self.transform(x)

        return x, y


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
