import torch
from torch import nn


def fed_avg(models, weights):
    # Get the state_dicts of the models
    state_dicts = [model.state_dict() for model in models]

    # Aggregate
    avg_state_dict = {key: torch.zeros_like(val) for key, val in state_dicts[0].items()}

    for state_dict, weight in zip(state_dicts, weights):
        for key in avg_state_dict:
            if "num_batches_tracked" in key:
                continue
            avg_state_dict[key] += state_dict[key] * weight

    for model in models:
        model.load_state_dict(avg_state_dict)

    return models


def grad_avg(models, active_indices, weights=None):
    # If weights not provided, use uniform weighting
    if weights is None:
        weights = [1] * len(active_indices)

    # Get the gradients of the models (they remain on their current devices)
    grads = [get_gradients(models[i]) for i in active_indices]

    # Aggregate gradients: sum weight-adjusted gradients for each parameter index
    avg_grads = [
        sum(grads[i][j] * weights[i] for i in range(len(grads)))
        for j in range(len(grads[0]))
    ]

    for model in models:
        set_gradients(avg_grads, model)

    return models


def get_gradients(model):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.clone())
        else:
            grads.append(torch.zeros_like(param))
    return grads


def set_gradients(grads, model):
    for param, grad in zip(model.parameters(), grads):
        param.grad = grad


def replace_bn_with_gn(model, num_groups=32):
    """
    Recursively replaces all BatchNorm layers in the given model with GroupNorm layers.

    Args:
        model (torch.nn.Module): The input PyTorch model.
        num_groups (int): The number of groups for GroupNorm layers.

    Returns:
        torch.nn.Module: The modified model with GroupNorm layers.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm1d):
            gn = nn.GroupNorm(num_groups, module.num_features)
            setattr(model, name, gn)
        elif isinstance(module, nn.BatchNorm2d):
            gn = nn.GroupNorm(num_groups, module.num_features)
            setattr(model, name, gn)
        elif isinstance(module, nn.BatchNorm3d):
            gn = nn.GroupNorm(num_groups, module.num_features)
            setattr(model, name, gn)
        else:
            replace_bn_with_gn(module, num_groups)

    return model
