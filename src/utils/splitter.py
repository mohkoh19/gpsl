import numpy as np
from scipy.stats import dirichlet
from torch import nn


def split_model(model, cut_layer):
    client_model = nn.Sequential()
    server_model = model

    dfs_split(server_model, cut_layer, "", client_model, replace_layer=nn.Identity)

    return client_model, server_model


def dfs_split(module, cut_layer, parent_name, sequential, replace_layer=None):
    """
    Recursively apply DFS to a module to find a specific layer.
    Replaces layers along the way if a replacement layer is specified.
    Adds layers along the way to a sequential container.
    """

    for name, sub_module in module.named_children():
        # Construct the full name of the submodule
        full_name = f"{parent_name}.{name}" if parent_name else name

        if full_name == cut_layer:
            return True

        # Add the submodule to the structure
        # structure.append((full_name, sub_module))

        # Recursively apply DFS to children and extend the structure list
        if list(sub_module.children()):
            child_sequential = nn.Sequential()
            sequential.add_module(name, child_sequential)
            if dfs_split(
                sub_module, cut_layer, full_name, child_sequential, replace_layer
            ):
                return True
        else:
            sequential.add_module(name, sub_module)
            if replace_layer is not None:
                setattr(module, name, replace_layer())


def replace_layers(model, layer_num_to_stop, new_layer_type=nn.Identity):
    """
    Recursively replace layers in the model up to layer_num_to_stop with new_layer_type.
    Also returns a nn.Sequential module containing all the replaced layers.
    """
    replaced_layers = nn.Sequential()

    def _replace_layers(module: nn.Module) -> nn.Module:
        nonlocal layer_num_to_stop
        if layer_num_to_stop <= 0:
            return module  # if we've replaced up to the desired layer, return original module
        for name, sub_module in module.named_children():
            if layer_num_to_stop > 0:
                if list(sub_module.children()):
                    for i, sub_sub_module in enumerate(sub_module):
                        if layer_num_to_stop > 0:
                            new_layer = _replace_layers(sub_sub_module)
                            if new_layer is not sub_sub_module:
                                replaced_layers.add_module(
                                    str(layer_num_to_stop) + "_" + name, sub_sub_module
                                )
                            sub_module[i] = new_layer
                else:
                    new_layer = _replace_layers(sub_module)
                    if new_layer is not sub_module:
                        replaced_layers.add_module(
                            str(layer_num_to_stop) + "_" + name, sub_module
                        )
                    setattr(module, name, new_layer)
        layer_num_to_stop -= 1
        return new_layer_type() if layer_num_to_stop >= 0 else module

    new_model = _replace_layers(model)
    return replaced_layers, new_model


def get_layer_num(model, target_name):
    """
    Recursively search for the target_name in the model and return the layer number.
    """
    layer_num = 1

    def _get_layer_num(module: nn.Module, name_prefix=""):
        nonlocal layer_num
        for name, sub_module in module.named_children():
            full_name = f"{name_prefix}{name}" if name_prefix else name
            if full_name == target_name:
                return layer_num
            if not list(sub_module.children()):
                layer_num += 1
            result = _get_layer_num(sub_module, f"{full_name}.")
            if result is not None:
                return result

    return _get_layer_num(model)


def split_dataset(
    Y: np.ndarray,
    num_clients: int,
    distribution: str = "iid",
    random_state: int = None,
    **kwargs,
) -> list:
    """
    Splits a dataset into sub-datasets for each client.

    Args:
        X (np.ndarray): Features of the dataset.
        Y (np.ndarray): Labels of the dataset, can be multi-class or multi-label.
        num_clients (int): Number of clients.
        distribution (str): Distribution strategy for allocating samples to clients.
            Options: "iid", "non_iid".
        random_state (int): Random seed for reproducibility.
        **kwargs: Additional arguments for the distribution strategy.

    Returns:
        list(list): List of index lists for each client.
    """
    Y = np.array(Y)

    if random_state:
        np.random.seed(random_state)

    if distribution == "iid":
        client_indices = uniform_allocation(Y, num_clients, random_state=random_state)

    elif distribution == "non_iid":
        client_indices = extended_dirichlet_allocation(
            Y, num_clients, random_state=random_state, **kwargs
        )

    return client_indices


def uniform_allocation(Y, num_clients, random_state=None):
    client_indices = []

    # Check if Y is multi-label
    multi_label = Y.ndim > 1 and Y.shape[1] > 1
    # X = np.arange(len(Y))

    if not multi_label:
        # sss = StratifiedShuffleSplit(n_splits=num_clients, random_state=random_state)
        # for _, test_index in sss.split(X, Y):
        #     client_indices.append(test_index.tolist())

        # Randomly shuffle indices
        indices = np.arange(len(Y))
        np.random.shuffle(indices)
        indices_split = np.array_split(indices, num_clients)
        client_indices = [list(idx) for idx in indices_split]
    else:
        indices = np.arange(len(Y))
        np.random.shuffle(indices)
        indices_split = np.array_split(indices, num_clients)
        client_indices = [list(idx) for idx in indices_split]

    return client_indices


def extended_dirichlet_allocation(Y, num_clients, C, alpha, random_state=None):
    """
    Perform extended Dirichlet allocation to partition the dataset among clients,
    ensuring each class is assigned to at least one client.

    Parameters
    ----------
    Y : array-like
        The class labels. If `multi_label` is False, it should be a 1D array with shape (n_samples,).
        If `multi_label` is True, it should be a 2D binary class label matrix with shape (n_samples, n_classes).
    num_clients : int
        The number of clients to distribute the data among.
    C : int
        The number of classes from which each client should have samples.
    alpha : float or array-like
        The concentration parameter(s) for the Dirichlet distribution.
    random_state : int, RandomState instance or None, optional
        The random seed for reproducibility. If None, the random state is not fixed.

    Returns
    -------
    list of lists
        A list containing lists of indices, where each sublist corresponds to the indices
        of samples allocated to a client.

    Notes
    -----
    This function uses a combination of multinomial and Dirichlet distributions to allocate
    samples to clients in a way that each class is guaranteed to be represented at least once.

    If `num_clients` * `C` is less than the number of classes, this function may not allocate
    every class to a client when `C` is 1.

    From: http://arxiv.org/abs/2302.01633

    Examples
    --------
    >>> Y = np.array([0, 1, 2, 0, 1, 2])
    >>> allocations = extended_dirichlet_allocation(Y, num_clients=3, C=1, alpha=0.5)
    >>> print(allocations)
    [[indices for client 1], [indices for client 2], [indices for client 3]]
    """
    np.random.seed(random_state)

    # Check if Y is multi-label. If so, randomly select one label for each sample.
    if Y.ndim > 1 and Y.shape[1] > 1:
        Y = np.array(
            [np.random.choice(np.where(Y[i] == 1)[0]) for i in range(Y.shape[0])]
        )

    # Determine the number of classes
    num_classes = np.unique(Y).size

    # Allocate classes to clients using a multinomial distribution
    # Each client can potentially have multiple of the same class
    q_C = np.zeros((num_clients, num_classes))
    for client in range(num_clients):
        choice_probs = np.exp(-q_C.sum(axis=0) * num_classes)
        choice_probs /= choice_probs.sum()
        class_choices = np.random.choice(
            num_classes, size=C, p=choice_probs, replace=False
        )
        q_C[client, class_choices] += 1

    # Allocate samples to clients
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        q_c = q_C[:, c] * alpha + 1e-10
        p_c = dirichlet.rvs(q_c, size=1).flatten()
        class_samples = np.where(Y == c)[0]
        sample_allocations = np.random.choice(
            range(num_clients), size=class_samples.size, p=p_c
        )
        for i in range(class_samples.size):
            client = sample_allocations[i]
            client_indices[client].append(class_samples[i])

    return client_indices
