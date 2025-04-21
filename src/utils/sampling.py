from collections.abc import Iterator
from copy import deepcopy

import numpy as np
from torch.utils.data import Dataset, RandomSampler
from torch.utils.data.sampler import Sampler

# from src.utils.lds import latent_dirichlet_sampling


class LocalBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: Dataset,
        drop_last=False,
    ) -> None:
        self.drop_last = drop_last

        self.sampler = RandomSampler(data_source=dataset)

        self.batch_sizes = None

        self.dataset_length = len(dataset)

        Y = [dataset[i][1] for i in range(self.dataset_length)]
        self.label_counts = np.bincount(Y)

    def get_dataset_length(self):
        return self.dataset_length

    def get_label_counts(self):
        return self.label_counts

    def set_batch_sizes(self, batch_sizes):
        self.batch_sizes = batch_sizes

    def reset(self):
        self.batch_sizes = None

    def __len__(self) -> int:
        return len(self.batch_sizes)

    def __iter__(self) -> Iterator[list[int]]:
        iterator = []

        sampler_iter = iter(self.sampler)
        iterator = [
            [next(sampler_iter) for _ in range(batch_size)]
            for batch_size in self.batch_sizes
        ]

        if self.drop_last:
            iterator = iterator[:-1]

        yield from iterator


class GlobalBatchSampler:
    def __init__(self, sampler, local_batch_samplers, batch_size):
        self.sampler = sampler(
            local_batch_samplers=local_batch_samplers, batch_size=batch_size
        )

        self.local_batch_samplers = local_batch_samplers

        self.client_sequences = []

        self.generate_batches()

    def update_local_batch_samplers(self, batches):
        batch_client_matrix = np.array(batches)  # B x K

        # For each client, get the corresponding column
        # Remove all zeros and convert to list
        # Send to remote
        for i, local_batch_sampler in enumerate(self.local_batch_samplers):
            client_batch = batch_client_matrix[:, i]
            client_batch = client_batch[client_batch != 0].tolist()
            local_batch_sampler.set_batch_sizes(client_batch)

    def get_client_sequence(self, batches):
        batch_client_matrix = np.array(batches)  # B x K

        # For each batch, get the corresponding row
        # Replace all values with the corresponding client index
        # Remove all zeros and convert to list of lists
        client_sequences = [
            np.where(batch)[0].tolist() for batch in batch_client_matrix
        ]
        return client_sequences

    def generate_batches(self, *args, **kwargs):
        for local_batch_sampler in self.local_batch_samplers:
            local_batch_sampler.reset()

        batches = self.sampler.generate_batches(*args, **kwargs)
        self.update_local_batch_samplers(batches)
        self.client_sequences = self.get_client_sequence(batches)

    def __len__(self):
        return len(self.client_sequences)

    def __getitem__(self, index):
        if not self.client_sequences:
            raise Exception(
                "GlobalBatchSampler: generate_batches() has not been called; no batches available."
            )
        return self.client_sequences[index]

    def __iter__(self):
        if not self.client_sequences:
            raise Exception(
                "GlobalBatchSampler: generate_batches() has not been called; no batches available."
            )
        yield from self.client_sequences


class LocalFixedSampler:
    def __init__(self, local_batch_samplers, batch_size, proportional=True) -> None:
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
        batches = []
        N = np.array(self.dataset_lengths)

        num_depleted = 0
        local_batch_sizes = self.local_batch_sizes.copy()

        while sum(N) > 0:
            b = np.zeros(self.num_clients)
            for i in range(self.num_clients):
                lbs = local_batch_sizes[i]

                if lbs == 0:
                    continue

                num_samples = min(lbs, N[i])

                b[i] += num_samples
                N[i] -= num_samples

                # If the batch is empty, update the local batch size
                if np.sum(N == 0) > num_depleted:
                    num_depleted = np.sum(N == 0)
                    local_batch_sizes[i] = 0

            batches.append(b.astype(int))

        return batches


class LocalFixedSamplerNonProportional(LocalFixedSampler):
    def __init__(self, local_batch_samplers, batch_size, proportional=True) -> None:
        super().__init__(local_batch_samplers, batch_size, proportional=False)


class UniformGlobalSampler:
    def __init__(self, local_batch_samplers, batch_size) -> None:
        self.batch_size = batch_size
        dataset_lengths = [ls.get_dataset_length() for ls in local_batch_samplers]
        self.samples = []
        for i, length in enumerate(dataset_lengths):
            self.samples += [i] * length
        self.num_clients = len(local_batch_samplers)

    def generate_batches(self, *args, **kwargs):
        # Randomly shuffle the superset
        np.random.shuffle(self.samples)

        num_batches = np.ceil(len(self.samples) / self.batch_size).astype(int)
        batches = np.array_split(np.array(self.samples), num_batches)

        # Count each client in each batch
        batches = [np.bincount(batch, minlength=self.num_clients) for batch in batches]

        return batches


# class LatentDirichletSampler:
#     def __init__(
#         self,
#         local_batch_samplers,
#         batch_size,
#         c=1.0,
#         sample_size=1000,
#         reinit=False,
#         tol=1e-5,
#         max_iter=1000,
#         delta=0.0,
#     ) -> None:
#         self.batch_size = batch_size
#         self.reinit = reinit
#         self.N = np.array([ls.get_dataset_length() for ls in local_batch_samplers])
#         self.delta = delta

#         N_total = self.N.sum()
#         r = self.N / N_total
#         Z = (sample_size * (N_total - 1)) / (N_total - sample_size) - 1
#         alpha = [
#             r_m * c * Z for r_m in r
#         ]  # Higher c decreases the variance of the distribution

#         client_label_counts = [
#             local_batch_sampler.get_label_counts()
#             for local_batch_sampler in local_batch_samplers
#         ]

#         # Fill the zeros since bincount did not receive a minimum length
#         num_classes = max([len(label_count) for label_count in client_label_counts])
#         client_label_counts = [
#             np.pad(label_count, (0, num_classes - len(label_count)))
#             for label_count in client_label_counts
#         ]
#         client_label_counts = np.array(client_label_counts)

#         beta = client_label_counts / np.sum(client_label_counts, axis=1, keepdims=True)

#         total_label_counts = client_label_counts.sum(axis=0)
#         num_classes = len(total_label_counts)

#         X = np.repeat(np.arange(num_classes), total_label_counts)
#         self.X = np.eye(num_classes)[X]
#         np.random.shuffle(self.X)
#         # X = np.split(X, range(batch_size, len(X), batch_size))

#         self.lds_params = {
#             "X": self.X[:sample_size],
#             "beta": beta,
#             "alpha": alpha,
#             "tol": tol,
#             "max_iter": max_iter,
#             "mle": False,
#             "pi": None,
#         }

#     def generate_batches(self, delays):
#         np.random.shuffle(self.X)
#         lds_params = deepcopy(self.lds_params)
#         lds_params["X"] = self.X[: len(lds_params["X"])]
#         print("delays: ", delays)
#         print("alpha before: ", lds_params["alpha"])
#         if self.delta > 0:
#             d_times = np.array(delays)

#             if np.std(d_times) > 0:
#                 sigma = (d_times - np.mean(d_times)) / (np.std(d_times))
#                 alpha = lds_params["alpha"]
#                 # alpha = [
#                 #     alpha[i] + alpha[i] * self.delta * np.exp(self.delta * d_times[i])
#                 #     for i in range(len(alpha))
#                 # ]
#                 alpha = [
#                     alpha[i] * np.exp(self.delta * sigma[i]) for i in range(len(alpha))
#                 ]
#                 lds_params["alpha"] = alpha

#         print("alpha after: ", lds_params["alpha"])

#         batches = latent_dirichlet_sampling(
#             lds_params, self.N.copy(), self.batch_size, reinit=self.reinit
#         )

#         return batches
