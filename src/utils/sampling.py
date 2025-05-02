from collections.abc import Iterator

import numpy as np
from torch.utils.data import Dataset, RandomSampler
from torch.utils.data.sampler import Sampler


class LocalBatchSampler(Sampler[list[int]]):
    """
    A batch sampler for individual clients. Samples batches using a RandomSampler
    and supports dynamic per-batch sizing.
    """

    def __init__(self, dataset: Dataset, drop_last=False) -> None:
        """
        Args:
            dataset (Dataset): The dataset to sample from.
            drop_last (bool): Whether to drop the last incomplete batch.
        """
        self.drop_last = drop_last
        self.sampler = RandomSampler(data_source=dataset)
        self.batch_sizes = None
        self.dataset_length = len(dataset)
        Y = [dataset[i][1] for i in range(self.dataset_length)]
        self.label_counts = np.bincount(Y)

    def get_dataset_length(self):
        """Returns the length of the dataset."""
        return self.dataset_length

    def get_label_counts(self):
        """Returns the label counts for the dataset."""
        return self.label_counts

    def set_batch_sizes(self, batch_sizes):
        """
        Sets the batch sizes to use during iteration.
        Args:
            batch_sizes (list[int]): Batch sizes for each batch.
        """
        self.batch_sizes = batch_sizes

    def reset(self):
        """Clears any existing batch size schedule."""
        self.batch_sizes = None

    def __len__(self) -> int:
        """Returns the number of batches."""
        return len(self.batch_sizes)

    def __iter__(self) -> Iterator[list[int]]:
        """
        Yields:
            Lists of indices for each batch according to the defined batch sizes.
        """
        sampler_iter = iter(self.sampler)
        iterator = [
            [next(sampler_iter) for _ in range(batch_size)]
            for batch_size in self.batch_sizes
        ]
        if self.drop_last:
            iterator = iterator[:-1]
        yield from iterator


class GlobalBatchSampler:
    """
    A sampler that coordinates batch sampling across multiple clients.
    """

    def __init__(self, sampler, local_batch_samplers, batch_size):
        """
        Args:
            sampler: A batch allocation strategy object with a `generate_batches` method.
            local_batch_samplers (list): List of LocalBatchSampler instances.
            batch_size (int): Global batch size.
        """
        self.sampler = sampler(
            local_batch_samplers=local_batch_samplers, batch_size=batch_size
        )
        self.local_batch_samplers = local_batch_samplers
        self.client_sequences = []
        self.generate_batches()

    def update_local_batch_samplers(self, batches):
        """
        Updates each LocalBatchSampler with its allocated batch sizes.
        """
        batch_client_matrix = np.array(batches)
        for i, local_batch_sampler in enumerate(self.local_batch_samplers):
            client_batch = batch_client_matrix[:, i]
            client_batch = client_batch[client_batch != 0].tolist()
            local_batch_sampler.set_batch_sizes(client_batch)

    def get_client_sequence(self, batches):
        """
        Constructs a per-batch list of participating client indices.
        """
        batch_client_matrix = np.array(batches)
        return [np.where(batch)[0].tolist() for batch in batch_client_matrix]

    def generate_batches(self, *args, **kwargs):
        """
        Generates new batch allocations and updates the local samplers.
        """
        for local_batch_sampler in self.local_batch_samplers:
            local_batch_sampler.reset()
        batches = self.sampler.generate_batches(*args, **kwargs)
        self.update_local_batch_samplers(batches)
        self.client_sequences = self.get_client_sequence(batches)

    def __len__(self):
        """Returns the number of batches."""
        return len(self.client_sequences)

    def __getitem__(self, index):
        """
        Returns the client sequence for the batch at the given index.
        Raises:
            Exception: If generate_batches has not been called.
        """
        if not self.client_sequences:
            raise Exception(
                "GlobalBatchSampler: generate_batches() has not been called; no batches available."
            )
        return self.client_sequences[index]

    def __iter__(self):
        """
        Iterates over the client sequences for each batch.
        Raises:
            Exception: If generate_batches has not been called.
        """
        if not self.client_sequences:
            raise Exception(
                "GlobalBatchSampler: generate_batches() has not been called; no batches available."
            )
        yield from self.client_sequences


class LocalFixedSampler:
    """
    Allocates fixed local batch sizes to each client based on their dataset size,
    either proportionally or uniformly.
    """

    def __init__(self, local_batch_samplers, batch_size, proportional=True) -> None:
        """
        Args:
            local_batch_samplers (list): List of LocalBatchSampler instances.
            batch_size (int): Global batch size.
            proportional (bool): If True, allocate batch sizes proportionally to dataset sizes. If False, allocate uniformly.
        """
        self.batch_size = batch_size
        self.dataset_lengths = [ls.get_dataset_length() for ls in local_batch_samplers]
        self.total_samples = sum(self.dataset_lengths)
        self.num_clients = len(local_batch_samplers)
        if proportional:
            self.local_batch_sizes = np.round(
                [
                    batch_size * (length / self.total_samples)
                    for length in self.dataset_lengths
                ]
            ).astype(int)
        else:
            self.local_batch_sizes = np.round(
                [batch_size / self.num_clients] * self.num_clients
            ).astype(int)
        self.local_batch_sizes[self.local_batch_sizes == 0] = 1

    def generate_batches(self, *args, **kwargs):
        """
        Generates batches as a list of client contribution vectors.
        Returns:
            List of np.ndarray with client-wise sample counts per batch.
        """
        batches = []
        N = np.array(self.dataset_lengths)
        local_batch_sizes = self.local_batch_sizes.copy()
        num_depleted = 0
        while sum(N) > 0:
            b = np.zeros(self.num_clients)
            for i in range(self.num_clients):
                lbs = local_batch_sizes[i]
                if lbs == 0:
                    continue
                num_samples = min(lbs, N[i])
                b[i] += num_samples
                N[i] -= num_samples
                if np.sum(N == 0) > num_depleted:
                    num_depleted = np.sum(N == 0)
                    local_batch_sizes[i] = 0
            batches.append(b.astype(int))
        return batches


class LocalFixedSamplerNonProportional(LocalFixedSampler):
    """
    A non-proportional fixed sampler where each client contributes an equal batch size.
    """

    def __init__(self, local_batch_samplers, batch_size, proportional=True) -> None:
        """
        Args:
            local_batch_samplers (list): List of LocalBatchSampler instances.
            batch_size (int): Global batch size.
            proportional (bool): Ignored, always uses non-proportional allocation.
        """
        super().__init__(local_batch_samplers, batch_size, proportional=False)


class UniformGlobalSampler:
    """
    Samples client contributions globally and uniformly across all clients.
    """

    def __init__(self, local_batch_samplers, batch_size) -> None:
        """
        Args:
            local_batch_samplers (list): List of LocalBatchSampler instances.
            batch_size (int): Global batch size.
        """
        self.batch_size = batch_size
        dataset_lengths = [ls.get_dataset_length() for ls in local_batch_samplers]
        self.samples = [
            i for i, length in enumerate(dataset_lengths) for _ in range(length)
        ]
        self.num_clients = len(local_batch_samplers)

    def generate_batches(self, *args, **kwargs):
        """
        Uniformly samples batches from the union of all clients' data.

        Returns:
            List of np.ndarrays representing client-wise sample counts per batch.
        """
        np.random.shuffle(self.samples)
        num_batches = np.ceil(len(self.samples) / self.batch_size).astype(int)
        batches = np.array_split(np.array(self.samples), num_batches)
        return [np.bincount(batch, minlength=self.num_clients) for batch in batches]
