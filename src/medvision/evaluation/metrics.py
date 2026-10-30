"""Classification metrics over the canonical HAM10000 classes.

All metrics are computed from the full set of predictions for an epoch,
using every class in ``CLASS_NAMES`` (in canonical order), even when a class
is absent from the data.

Definitions:

- Per-class precision is 0 when the class was never predicted; recall is 0
  when the class has no support; F1 is 0 when precision + recall is 0.
- Macro precision, recall, and F1 are unweighted means over all classes,
  including classes with no support (which contribute 0).
- Balanced accuracy is the mean recall over classes with support > 0 only.
  It therefore equals macro recall when every class is present, and is
  higher than macro recall when some classes are absent.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from medvision.data.constants import CLASS_NAMES


def compute_confusion_matrix(
    targets: Sequence[int],
    predictions: Sequence[int],
    num_classes: int = len(CLASS_NAMES),
) -> np.ndarray:
    """Return a ``num_classes x num_classes`` matrix (rows true, columns predicted)."""

    targets_array = np.asarray(targets, dtype=int)
    predictions_array = np.asarray(predictions, dtype=int)

    if targets_array.shape != predictions_array.shape or targets_array.ndim != 1:
        raise ValueError("targets and predictions must be 1-D sequences of equal length.")

    if targets_array.size == 0:
        raise ValueError("Cannot compute metrics without samples.")

    for name, values in (("targets", targets_array), ("predictions", predictions_array)):
        if values.min() < 0 or values.max() >= num_classes:
            raise ValueError(f"{name} contain class indices outside 0..{num_classes - 1}.")

    matrix = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(matrix, (targets_array, predictions_array), 1)

    return matrix


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def compute_classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict[str, Any]:
    """Compute overall, macro, and per-class metrics."""

    matrix = compute_confusion_matrix(targets, predictions, len(class_names))

    true_positives = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)

    precision = _safe_divide(true_positives, predicted)
    recall = _safe_divide(true_positives, support)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return {
        "sample_count": int(matrix.sum()),
        "accuracy": float(true_positives.sum() / matrix.sum()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
        "confusion_matrix": matrix.tolist(),
    }


def confusion_matrix_frame(
    matrix: Sequence[Sequence[int]],
    class_names: Sequence[str] = CLASS_NAMES,
) -> pd.DataFrame:
    """Label a confusion matrix with true (rows) and predicted (columns) classes."""

    return pd.DataFrame(
        matrix,
        index=pd.Index([f"true_{name}" for name in class_names], name="true_class"),
        columns=[f"pred_{name}" for name in class_names],
    )
