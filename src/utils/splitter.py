import numpy as np
from scipy.stats import dirichlet


def split_model(model, cut_layer):
    """
    Splits a model into a client-side and server-side model at a specified cut layer.

    Parameters
    ----------
    model : torch.nn.Module
        The complete model to split.
    cut_layer : str
        The name of the layer at which to split the model.

    Returns
    -------
    tuple
        A tuple containing the client model (torch.nn.Sequential) and the modified server model.
    """
    import torch.nn as nn

    client_model = nn.Sequential()
    server_model = model

    dfs_split(server_model, cut_layer, "", client_model, replace_layer=nn.Identity)
    return client_model, server_model


def dfs_split(module, cut_layer, parent_name, sequential, replace_layer=None):
    """
    Recursively traverses the model to extract layers up to the cut layer into a Sequential block.

    Parameters
    ----------
    module : torch.nn.Module
        The current module to traverse.
    cut_layer : str
        The full name of the target cut layer.
    parent_name : str
        Name prefix for building full layer names.
    sequential : torch.nn.Sequential
        Sequential container to hold extracted layers.
    replace_layer : torch.nn.Module, optional
        If provided, replaces layers after the cut point with this layer.
    """
    import torch.nn as nn

    for name, sub_module in module.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        if full_name == cut_layer:
            return True
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


def replace_layers(model, layer_num_to_stop, new_layer_type=None):
    """
    Replaces a fixed number of layers in the model with a specified layer type.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be modified.
    layer_num_to_stop : int
        Number of layers to replace from the start.
    new_layer_type : torch.nn.Module, optional
        Layer to replace with, defaults to Identity.

    Returns
    -------
    tuple
        A tuple of (replaced_layers: nn.Sequential, new_model: nn.Module).
    """
    import torch.nn as nn

    if new_layer_type is None:
        new_layer_type = nn.Identity

    replaced_layers = nn.Sequential()

    def _replace_layers(module: nn.Module) -> nn.Module:
        nonlocal layer_num_to_stop
        if layer_num_to_stop <= 0:
            return module
        for name, sub_module in module.named_children():
            if layer_num_to_stop > 0:
                if list(sub_module.children()):
                    for i, sub_sub_module in enumerate(sub_module):
                        if layer_num_to_stop > 0:
                            new_layer = _replace_layers(sub_sub_module)
                            if new_layer is not sub_sub_module:
                                replaced_layers.add_module(
                                    f"{layer_num_to_stop}_{name}", sub_sub_module
                                )
                            sub_module[i] = new_layer
                else:
                    new_layer = _replace_layers(sub_module)
                    if new_layer is not sub_module:
                        replaced_layers.add_module(
                            f"{layer_num_to_stop}_{name}", sub_module
                        )
                    setattr(module, name, new_layer)
        layer_num_to_stop -= 1
        return new_layer_type() if layer_num_to_stop >= 0 else module

    new_model = _replace_layers(model)
    return replaced_layers, new_model


def get_layer_num(model, target_name):
    """
    Finds the layer number of a named layer in a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to search.
    target_name : str
        Full name of the target layer.

    Returns
    -------
    int or None
        Index of the layer, or None if not found.
    """
    layer_num = 1

    def _get_layer_num(module, name_prefix=""):
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


def split_dataset(Y, num_clients, distribution="iid", random_state=None, **kwargs):
    """
    Splits dataset indices among clients using specified distribution.

    Parameters
    ----------
    Y : np.ndarray
        Label array (1D or multi-label).
    num_clients : int
        Number of clients.
    distribution : str, optional
        Distribution type: 'iid' or 'non_iid'.
    random_state : int, optional
        Random seed for reproducibility.
    **kwargs
        Additional arguments for non-IID splitting (e.g., C, alpha).

    Returns
    -------
    list of lists
        List of index lists for each client.
    """
    Y = np.array(Y)
    if random_state is not None:
        np.random.seed(random_state)

    if distribution == "iid":
        return uniform_allocation(Y, num_clients, random_state=random_state)
    elif distribution == "non_iid":
        return extended_dirichlet_allocation(
            Y, num_clients, random_state=random_state, **kwargs
        )
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")


def uniform_allocation(Y, num_clients, random_state=None):
    """
    Uniformly allocate samples across clients.

    Parameters
    ----------
    Y : np.ndarray
        Label array.
    num_clients : int
        Number of clients.
    random_state : int, optional
        Random seed.

    Returns
    -------
    list of lists
        Client-wise list of sample indices.
    """
    indices = np.arange(len(Y))
    np.random.shuffle(indices)
    indices_split = np.array_split(indices, num_clients)
    return [list(idx) for idx in indices_split]


def extended_dirichlet_allocation(Y, num_clients, C, alpha, random_state=None):
    """
    Perform extended Dirichlet allocation to partition the dataset among clients,
    ensuring each class is assigned to at least one client.

    Parameters
    ----------
    Y : array-like
        Class labels (1D) or multi-label matrix (2D).
    num_clients : int
        Number of clients.
    C : int
        Number of classes per client.
    alpha : float
        Dirichlet concentration parameter.
    random_state : int, optional
        Random seed.

    Returns
    -------
    list of lists
        Client-wise index allocation.

    References
    ----------
    https://arxiv.org/abs/2302.01633
    """
    # Convert multi-label to single-class if necessary
    if Y.ndim > 1 and Y.shape[1] > 1:
        Y = np.array(
            [np.random.choice(np.where(Y[i] == 1)[0]) for i in range(Y.shape[0])]
        )

    num_classes = np.unique(Y).size

    q_C = np.zeros((num_clients, num_classes))
    for client in range(num_clients):
        choice_probs = np.exp(-q_C.sum(axis=0) * num_classes)
        choice_probs /= choice_probs.sum()
        class_choices = np.random.choice(
            num_classes, size=C, p=choice_probs, replace=False
        )
        q_C[client, class_choices] += 1

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        q_c = q_C[:, c] * alpha + 1e-10
        p_c = dirichlet.rvs(q_c, size=1).flatten()
        class_samples = np.where(Y == c)[0]
        sample_allocations = np.random.choice(
            range(num_clients), size=class_samples.size, p=p_c
        )
        for i, client in enumerate(sample_allocations):
            client_indices[client].append(class_samples[i])

    return client_indices
