import numpy as np

from utils.cmm import iterative_em_clustering


## Clustering functions
def em_clustering(client_rrefs, cmm_sampler, **kwargs):
    subset_lengths = [
        client_rref.rpc_sync().get_dataset_length("train")
        for client_rref in client_rrefs
    ]

    target_distribution = cmm_sampler.target_distribution
    beta = cmm_sampler.label_distributions

    clusters = iterative_em_clustering(
        subset_lengths=subset_lengths,
        target_distribution=target_distribution,
        beta=beta,
        **kwargs,
    )

    clusters = [[client_rrefs[int(i)] for i in cluster] for cluster in clusters]
    return clusters


def naive_clustering(client_rrefs, num_clusters, **kwargs):
    subset_indices = np.arange(len(client_rrefs))

    np.random.shuffle(subset_indices)
    clusters = [[] for _ in range(num_clusters)]

    next_subset = 0
    for k in subset_indices:
        clusters[next_subset].append(k)
        next_subset = (next_subset + 1) % num_clusters

    clusters = [[client_rrefs[int(i)] for i in cluster] for cluster in clusters]

    return clusters
