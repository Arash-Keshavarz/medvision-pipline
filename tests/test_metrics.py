"""Tests for classification metrics."""

import pytest

from medvision.data.constants import CLASS_NAMES
from medvision.evaluation.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    confusion_matrix_frame,
)

# Classes 0-2 present; classes 3-6 absent from targets and predictions.
TARGETS = [0, 0, 1, 1, 2, 2]
PREDICTIONS = [0, 1, 1, 1, 2, 0]


def test_metrics_match_hand_calculation() -> None:
    metrics = compute_classification_metrics(TARGETS, PREDICTIONS)
    per_class = metrics["per_class"]

    # akiec: 1 of 2 predicted correct, 1 of 2 found.
    assert per_class["akiec"] == pytest.approx(
        {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}
    )
    # bcc: 2 of 3 predictions correct, both found.
    assert per_class["bcc"] == pytest.approx(
        {"precision": 2 / 3, "recall": 1.0, "f1": 0.8, "support": 2}
    )
    # bkl: its single prediction is correct, 1 of 2 found.
    assert per_class["bkl"] == pytest.approx(
        {"precision": 1.0, "recall": 0.5, "f1": 2 / 3, "support": 2}
    )

    assert metrics["sample_count"] == 6
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    # Means over all seven classes; absent classes contribute 0.
    assert metrics["macro_precision"] == pytest.approx((0.5 + 2 / 3 + 1.0) / 7)
    assert metrics["macro_recall"] == pytest.approx((0.5 + 1.0 + 0.5) / 7)
    assert metrics["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 7)
    # Balanced accuracy averages recall over the three classes with support.
    assert metrics["balanced_accuracy"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)


def test_all_canonical_classes_are_reported_in_order() -> None:
    metrics = compute_classification_metrics(TARGETS, PREDICTIONS)

    assert tuple(metrics["per_class"]) == CLASS_NAMES
    assert len(metrics["confusion_matrix"]) == len(CLASS_NAMES)

    for name in CLASS_NAMES[3:]:
        assert metrics["per_class"][name] == {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "support": 0,
        }


def test_prediction_of_absent_class_counts_as_zero_precision() -> None:
    metrics = compute_classification_metrics([0, 0], [0, 3])

    assert metrics["per_class"]["df"]["precision"] == 0.0
    assert metrics["per_class"]["df"]["support"] == 0
    # Only akiec has support, so balanced accuracy is its recall.
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["macro_recall"] == pytest.approx(0.5 / 7)


def test_perfect_predictions_with_all_classes() -> None:
    targets = list(range(len(CLASS_NAMES)))
    metrics = compute_classification_metrics(targets, targets)

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_confusion_matrix_rows_are_true_classes() -> None:
    matrix = compute_confusion_matrix(TARGETS, PREDICTIONS)

    assert matrix[0].tolist()[:3] == [1, 1, 0]
    assert matrix[1].tolist()[:3] == [0, 2, 0]
    assert matrix[2].tolist()[:3] == [1, 0, 1]
    assert matrix.sum() == len(TARGETS)

    frame = confusion_matrix_frame(matrix)

    assert frame.loc["true_akiec", "pred_bcc"] == 1
    assert list(frame.columns) == [f"pred_{name}" for name in CLASS_NAMES]


@pytest.mark.parametrize(
    ("targets", "predictions", "message"),
    [
        ([0, 1], [0], "equal length"),
        ([], [], "without samples"),
        ([0, 7], [0, 0], "outside"),
        ([0, 0], [0, -1], "outside"),
    ],
)
def test_invalid_inputs_fail(targets: list[int], predictions: list[int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compute_classification_metrics(targets, predictions)
