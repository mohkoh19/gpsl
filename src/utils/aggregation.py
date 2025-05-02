"""
Utility functions for gradient averaging and normalization layer replacement
used in Global Parallel Split Learning (GPSL).
"""

import torch
from torch import nn


def grad_avg(models, active_indices, weights=None):
    """
    Perform weighted gradient averaging over a subset of models.

    Parameters
    ----------
    models : list of torch.nn.Module
        List of models to update.
    active_indices : list of int
        Indices of models participating in the gradient averaging.
    weights : list of float, optional
        Weights for each participating model. If None, uniform weights are used.

    Returns
    -------
    list of torch.nn.Module
        Updated models with averaged gradients.
    """
    if weights is None:
        weights = [1] * len(active_indices)

    grads = [get_gradients(models[i]) for i in active_indices]

    avg_grads = [
        sum(grads[i][j] * weights[i] for i in range(len(grads)))
        for j in range(len(grads[0]))
    ]

    for model in models:
        set_gradients(avg_grads, model)

    return models


def get_gradients(model):
    """
    Extract gradients from a model's parameters.

    Parameters
    ----------
    model : torch.nn.Module
        Model from which to extract gradients.

    Returns
    -------
    list of torch.Tensor
        List of parameter gradients.
    """
    grads = []
    for param in model.parameters():
        grads.append(
            param.grad.clone() if param.grad is not None else torch.zeros_like(param)
        )
    return grads


def set_gradients(grads, model):
    """
    Assign gradients to a model's parameters.

    Parameters
    ----------
    grads : list of torch.Tensor
        Gradients to assign.
    model : torch.nn.Module
        Target model.
    """
    for param, grad in zip(model.parameters(), grads):
        param.grad = grad


def replace_bn_with_gn(model, num_groups=32):
    """
    Recursively replaces all BatchNorm layers in a model with GroupNorm layers.

    Parameters
    ----------
    model : torch.nn.Module
        Model in which to replace BatchNorm layers.
    num_groups : int
        Number of groups to use for GroupNorm.

    Returns
    -------
    torch.nn.Module
        Model with GroupNorm layers replacing BatchNorm layers.
    """
    for name, module in model.named_children():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            gn = nn.GroupNorm(num_groups, module.num_features)
            setattr(model, name, gn)
        else:
            replace_bn_with_gn(module, num_groups)
    return model
