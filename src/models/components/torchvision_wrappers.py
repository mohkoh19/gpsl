from torch import nn
import torchvision.models as models


def resnet18(num_classes, small_imgs=True, **kwargs):
    """
    Returns a modified ResNet-18 model.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    small_imgs : bool, optional
        If True, modifies the model for small images (e.g., CIFAR-10).
        This replaces the initial convolution with a 3x3 kernel and removes the max pooling layer.
    **kwargs : dict
        Additional keyword arguments passed to torchvision.models.resnet18.

    Returns
    -------
    torch.nn.Module
        Modified ResNet-18 model.
    """
    model = models.resnet18(**kwargs)

    if small_imgs:
        model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()

    model.fc = nn.Linear(in_features=512, out_features=num_classes)
    return model
