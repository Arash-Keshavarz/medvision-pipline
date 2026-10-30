"""ResNet18 classifier for HAM10000."""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from medvision.data.constants import NUM_CLASSES

PRETRAINED_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1


def build_resnet18(
    pretrained: bool = True,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """Build a ResNet18 whose final layer has ``num_classes`` outputs.

    With ``pretrained=True`` the ImageNet weights are loaded from the torch
    hub cache or downloaded. If that fails, an error is raised; the model
    never silently falls back to random initialisation. With
    ``pretrained=False`` no weights are loaded and no network access occurs.

    The model returns raw logits; apply softmax only when probabilities are
    needed.
    """

    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    weights = PRETRAINED_WEIGHTS if pretrained else None

    try:
        model = resnet18(weights=weights)
    except Exception as error:
        if weights is None:
            raise
        raise RuntimeError(
            f"Could not load pretrained ResNet18 weights ({weights}). Check network "
            "access or the torch hub cache, or set model.weights to 'none' for an "
            "explicitly randomly initialised model."
        ) from error

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
