import os
from torch.utils.data import Dataset
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
