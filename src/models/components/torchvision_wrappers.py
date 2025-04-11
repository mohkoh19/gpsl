import torchvision.models as models
from torch import nn


def resnet18(num_classes, small_imgs=True, alt=False, **kwargs):
    # original_model = models.resnet18(weights=ResNet18_Weights.DEFAULT, **kwargs)
    original_model = models.resnet18(**kwargs)

    if small_imgs:
        # Replace the first convolutional layer
        original_model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # Remove the max pooling layer
        original_model.maxpool = nn.Identity()

    # Replace the fully connected layer
    original_model.fc = nn.Linear(in_features=512, out_features=num_classes)

    return original_model


def alexnet(num_classes, **kwargs):
    model = models.alexnet(**kwargs)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model
