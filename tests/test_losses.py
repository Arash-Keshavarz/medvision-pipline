"""Tests for loss construction and training-manifest class weights."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from medvision.data.constants import CLASS_NAMES, NUM_CLASSES
from medvision.training.losses import (
    CLASS_WEIGHT_FORMULA,
    build_criterion,
    class_weights_from_manifest,
    compute_class_weights,
    weight_tensor,
)
from tests.conftest import SyntheticDataset

# Counts chosen so every weight differs: N = 28, weight = 28 / (7 * count).
COUNTS = {"akiec": 2, "bcc": 4, "bkl": 1, "df": 7, "mel": 4, "nv": 8, "vasc": 2}


def labels_from(counts: dict[str, int]) -> list[str]:
    # Reverse-alphabetical order, so canonical ordering cannot come from input order.
    return [name for name in sorted(counts, reverse=True) for _ in range(counts[name])]


def test_weights_follow_the_balanced_formula() -> None:
    result = compute_class_weights(labels_from(COUNTS))

    assert result["n_train"] == 28
    assert result["number_of_classes"] == NUM_CLASSES
    assert result["counts"] == COUNTS
    assert result["formula"] == CLASS_WEIGHT_FORMULA

    for name, count in COUNTS.items():
        assert result["weights"][name] == pytest.approx(28 / (7 * count))

    # Balanced weights restore equal total weight per class.
    totals = {name: result["weights"][name] * count for name, count in COUNTS.items()}
    assert all(total == pytest.approx(28 / 7) for total in totals.values())


def test_weights_are_in_canonical_class_order() -> None:
    result = compute_class_weights(labels_from(COUNTS))

    assert result["class_order"] == list(CLASS_NAMES)
    assert list(result["counts"]) == list(CLASS_NAMES)
    assert list(result["weights"]) == list(CLASS_NAMES)

    tensor = weight_tensor(result)
    for index, name in enumerate(CLASS_NAMES):
        assert tensor[index].item() == pytest.approx(28 / (7 * COUNTS[name]))


def test_missing_training_class_is_rejected() -> None:
    labels = [name for name in labels_from(COUNTS) if name != "df"]

    with pytest.raises(ValueError, match=r"no images of \['df'\]"):
        compute_class_weights(labels)


def test_unknown_class_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown classes"):
        compute_class_weights([*labels_from(COUNTS), "melanoma"])


def test_weights_use_only_the_training_manifest(synthetic_dataset: SyntheticDataset) -> None:
    train_path = synthetic_dataset.manifest_path("train")
    train_classes = pd.read_csv(train_path)["class_name"]

    result = class_weights_from_manifest(train_path)

    assert result["source_manifest"] == str(train_path)
    assert result["n_train"] == len(train_classes)
    assert result["counts"] == {name: int((train_classes == name).sum()) for name in CLASS_NAMES}


def test_non_training_manifest_is_rejected(synthetic_dataset: SyntheticDataset) -> None:
    with pytest.raises(ValueError, match="expected to contain only 'train'"):
        class_weights_from_manifest(synthetic_dataset.manifest_path("validation"))


def test_missing_class_in_training_manifest_is_rejected(
    synthetic_dataset: SyntheticDataset,
) -> None:
    path = synthetic_dataset.manifest_path("train")
    manifest = pd.read_csv(path)
    manifest[manifest["class_name"] != "vasc"].to_csv(path, index=False)

    with pytest.raises(ValueError, match=r"no images of \['vasc'\]"):
        class_weights_from_manifest(Path(path))


def test_plain_criterion_is_unweighted_cross_entropy() -> None:
    criterion = build_criterion("cross_entropy")

    assert isinstance(criterion, nn.CrossEntropyLoss)
    assert criterion.weight is None
    assert criterion.reduction == "mean"

    with pytest.raises(ValueError, match="does not take class weights"):
        build_criterion("cross_entropy", compute_class_weights(labels_from(COUNTS)))


def test_weighted_criterion_carries_canonical_weights() -> None:
    class_weights = compute_class_weights(labels_from(COUNTS))
    criterion = build_criterion("weighted_cross_entropy", class_weights)

    assert isinstance(criterion, nn.CrossEntropyLoss)
    assert torch.equal(criterion.weight, weight_tensor(class_weights))

    with pytest.raises(ValueError, match="requires class weights"):
        build_criterion("weighted_cross_entropy")

    with pytest.raises(ValueError, match="Unknown loss"):
        build_criterion("focal")
