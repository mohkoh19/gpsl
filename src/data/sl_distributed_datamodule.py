import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
from src.utils.splitter import split_dataset


class SubsetRandomSampler(torch.utils.data.Sampler):
    """A custom sampler that takes a list of Subsets and randomly samples from them.
    The difference is that it shuffles per subset.
    I.e., it iterates through the subsets from 0 to n and within each subset
    it shuffles the indices.
    """

    def __init__(self, subset_indices: list):
        # self.subset_indices = torch.tensor(self.subset_indices)
        self.subset_indices = subset_indices

    def __iter__(self):
        # Go through each subset and shuffle the indices
        shuffled_indices = []
        for subset in self.subset_indices:
            indices = subset.copy()
            indices = torch.tensor(indices)[torch.randperm(len(indices))].tolist()
            shuffled_indices.extend(indices)

        yield from shuffled_indices

    def __len__(self):
        return sum(len(subset) for subset in self.subset_indices)


class SubsetWrapper(Dataset):
    """A PyTorch Dataset that takes a list of Subsets and concatenates them.
    During iteration it adds the index of the subset to the sample.
    So the sample is a tuple (data, target, subset_index).
    """

    def __init__(self, subsets: list):
        self.subsets = subsets

        # Defines a lookup table that maps each idx to its subset index
        self.subset_indices = []
        for subset_index, subset in enumerate(subsets):
            self.subset_indices.extend([subset_index] * len(subset))
        self.subset_indices = torch.tensor(self.subset_indices)

        # Concatenate the datasets
        self.dataset = ConcatDataset(subsets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # ToDo: Avoid overlaps between subsets in batch sampling
        data, target = self.dataset[idx]
        subset_index = self.subset_indices[idx]
        return data, target, subset_index


class SLDistributedDataModule(LightningDataModule):
    def __init__(
        self,
        base_datamodule: LightningDataModule,
        num_clients: int,
        subset_config: dict,
    ):
        super().__init__()
        self.base = base_datamodule
        self.num_clients = num_clients
        self.num_classes = base_datamodule.num_classes
        self.subset_config = subset_config
        self.train_subset_dataloaders = None

        # ToDo: check what happens if we save hyperparameters

    def prepare_data(self):
        return self.base.prepare_data()

    def extract_loader_config(self, loader: DataLoader) -> dict:
        return {
            "num_workers": loader.num_workers,
            "pin_memory": loader.pin_memory,
            "timeout": loader.timeout,
            "worker_init_fn": loader.worker_init_fn,
            "persistent_workers": loader.persistent_workers,
            "batch_size": loader.batch_size,
        }

    def setup(self, stage=None):
        self.base.setup(stage)

        # Split the training dataset indices among clients
        train_dataset = self.base.train_dataloader().dataset

        # Retrieve targets from the dataset by iterating over the dataset
        targets = (
            [target for _, target in train_dataset]
            if hasattr(train_dataset, "__getitem__")
            else None
        )
        split_indices = split_dataset(
            targets,
            self.num_clients,
            self.subset_config["distribution"],
            **self.subset_config["params"],
        )

        subsets = [Subset(train_dataset, indices) for indices in split_indices]
        train_subset_lengths = [len(s) for s in subsets]
        self.client_weights = [
            sl / sum(train_subset_lengths) for sl in train_subset_lengths
        ]
        train_dataset = SubsetWrapper(subsets)

        # Create dataloaders for each subset
        base_loader = self.base.train_dataloader()
        loader_cfg = self.extract_loader_config(base_loader)

        self.train_loader = DataLoader(
            dataset=train_dataset,
            **loader_cfg,
            sampler=SubsetRandomSampler(split_indices),
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.base.val_dataloader()

    def teardown(self, stage=None):
        return self.base.teardown(stage)

    def state_dict(self):
        return self.base.state_dict()

    def load_state_dict(self, state_dict):
        return self.base.load_state_dict(state_dict)
