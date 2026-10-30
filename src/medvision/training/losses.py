"""Loss construction, including class weights from the training manifest.

Class weights use the "balanced" formula

    weight[c] = N_train / (NUM_CLASSES * count[c])

with counts taken from the training manifest only (never validation or test)
and weights ordered by ``CLASS_NAMES``. A class with no training images has no
finite weight, so it is rejected instead of being silently given one.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from medvision.data.constants import CLASS_NAMES, EXPECTED_CLASSES, NUM_CLASSES
from medvision.data.dataset import load_manifest

CLASS_WEIGHT_FORMULA = "weight[c] = N_train / (number_of_classes * count[c])"


def compute_class_weights(class_names: Iterable[str]) -> dict[str, Any]:
    """Return training class counts and balanced weights in canonical order."""

    labels = list(class_names)
    unknown = set(labels) - EXPECTED_CLASSES

    if unknown:
        raise ValueError(f"Unknown classes in training labels: {sorted(unknown)}")

    counts = {name: labels.count(name) for name in CLASS_NAMES}
    missing = [name for name, count in counts.items() if count == 0]

    if missing:
        raise ValueError(f"Cannot compute class weights: training data has no images of {missing}.")

    total = len(labels)
    weights = {name: total / (NUM_CLASSES * count) for name, count in counts.items()}

    return {
        "formula": CLASS_WEIGHT_FORMULA,
        "class_order": list(CLASS_NAMES),
        "number_of_classes": NUM_CLASSES,
        "n_train": total,
        "counts": counts,
        "weights": weights,
    }


def class_weights_from_manifest(manifest_path: Path) -> dict[str, Any]:
    """Compute class weights from a validated training manifest."""

    manifest = load_manifest(manifest_path, expected_split="train")
    return {"source_manifest": str(manifest_path), **compute_class_weights(manifest["class_name"])}


def weight_tensor(class_weights: dict[str, Any]) -> torch.Tensor:
    """Return the weights as a float tensor indexed by class index."""

    return torch.tensor([class_weights["weights"][name] for name in CLASS_NAMES])


def build_criterion(loss: str, class_weights: dict[str, Any] | None = None) -> nn.Module:
    """Build the training criterion named by the config's ``loss`` setting."""

    if loss == "cross_entropy":
        if class_weights is not None:
            raise ValueError("Plain cross_entropy does not take class weights.")
        return nn.CrossEntropyLoss()

    if loss == "weighted_cross_entropy":
        if class_weights is None:
            raise ValueError("weighted_cross_entropy requires class weights.")
        return nn.CrossEntropyLoss(weight=weight_tensor(class_weights))

    raise ValueError(f"Unknown loss {loss!r}.")
