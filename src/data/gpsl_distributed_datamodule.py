from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, Subset
from src.utils.splitter import split_dataset
from src.utils.sampling import GlobalBatchSampler, LocalBatchSampler


class DistributedDataModule(LightningDataModule):
    def __init__(
        self,
        base_datamodule: LightningDataModule,
        num_clients: int,
        subset_config: dict,
        sampler: object,
    ):
        super().__init__()
        self.base = base_datamodule
        self.num_clients = num_clients
        self.num_classes = base_datamodule.num_classes
        self.subset_config = subset_config
        self.train_subset_dataloaders = None
        self.sampler = sampler

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

        # Create dataloaders for each subset
        base_loader = self.base.train_dataloader()
        loader_cfg = self.extract_loader_config(base_loader)

        self.train_subset_dataloaders = [
            DataLoader(
                dataset=subset, **loader_cfg, batch_sampler=LocalBatchSampler(subset)
            )
            for subset in subsets
        ]

        # Create a global batch sampler to schedule batches among clients
        local_batch_samplers = [
            dl.batch_sampler for dl in self.train_subset_dataloaders
        ]

        self.global_batch_sampler = GlobalBatchSampler(
            self.sampler,
            local_batch_samplers,
            base_loader.batch_size,
        )

        self.global_batch_sampler.generate_batches()

    def train_dataloader(self):
        if self.train_subset_dataloaders is None:
            raise RuntimeError("setup() must be called before train_dataloader()")

        return CombinedLoader(
            self.train_subset_dataloaders,
            mode="max_size",
        )

    def on_train_epoch_start(self):
        self.global_batch_sampler.generate_batches()

    def val_dataloader(self):
        return self.base.val_dataloader()

    def teardown(self, stage=None):
        return self.base.teardown(stage)

    def state_dict(self):
        return self.base.state_dict()

    def load_state_dict(self, state_dict):
        return self.base.load_state_dict(state_dict)
