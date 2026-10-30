"""Epoch loops, best-score tracking, and checkpointing."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn

from medvision.data.constants import CLASS_NAMES, CLASS_TO_INDEX
from medvision.evaluation.metrics import compute_classification_metrics, confusion_matrix_frame


@dataclass(frozen=True)
class EpochResult:
    """Epoch loss and metrics over one pass of a loader.

    ``loss`` is the criterion's loss over the whole epoch, as if every sample
    had been in one batch (for weighted cross-entropy, the target-weighted
    mean). ``unweighted_cross_entropy`` is the plain per-sample mean
    cross-entropy, comparable across runs with different criteria.
    """

    loss: float
    unweighted_cross_entropy: float
    metrics: dict[str, Any]


def loss_normaliser(criterion: nn.Module, targets: torch.Tensor) -> float:
    """Return the denominator ``criterion`` uses for a mean-reduced batch loss.

    ``nn.CrossEntropyLoss(weight=w)`` with mean reduction divides the summed
    weighted losses by ``w[targets].sum()``, not by the batch size. Other
    mean-reduced criteria divide by the number of samples.
    """

    if getattr(criterion, "reduction", "mean") != "mean":
        raise ValueError("Epoch loss aggregation requires a mean-reduced criterion.")

    if isinstance(criterion, nn.CrossEntropyLoss) and criterion.weight is not None:
        if criterion.label_smoothing != 0:
            raise ValueError("Weighted cross-entropy with label smoothing is not supported.")
        return criterion.weight[targets].sum().item()

    return float(targets.shape[0])


def run_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochResult:
    """Run one training epoch (with ``optimizer``) or one evaluation pass.

    Each batch's mean loss is multiplied back by the criterion's denominator
    (see ``loss_normaliser``) and the epoch loss divides the total by the sum
    of denominators. This equals the criterion applied once to the
    concatenated epoch, so a smaller final batch, or a batch with a different
    class mix under weighted cross-entropy, is weighted correctly. Metrics are
    computed once from all predictions of the epoch.
    """

    training = optimizer is not None
    model.train(training)
    context = nullcontext() if training else torch.inference_mode()

    total_loss = 0.0
    total_normaliser = 0.0
    total_unweighted = 0.0
    sample_count = 0
    all_targets: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []

    with context:
        for batch_index, batch in enumerate(loader):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)

            if targets.ndim != 1 or logits.shape != (targets.shape[0], len(CLASS_NAMES)):
                raise ValueError(
                    f"Expected logits of shape ({targets.shape[0]}, {len(CLASS_NAMES)}) and "
                    f"1-D targets, got {tuple(logits.shape)} and {tuple(targets.shape)}."
                )

            loss = criterion(logits, targets)
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                raise RuntimeError(f"Non-finite loss {loss_value} at batch {batch_index}.")

            if training:
                loss.backward()
                optimizer.step()

            normaliser = loss_normaliser(criterion, targets)
            total_loss += loss_value * normaliser
            total_normaliser += normaliser
            total_unweighted += nn.functional.cross_entropy(
                logits.detach(), targets, reduction="sum"
            ).item()
            sample_count += targets.shape[0]
            all_targets.append(targets.detach().cpu())
            all_predictions.append(logits.detach().argmax(dim=1).cpu())

    if sample_count == 0:
        raise ValueError("Loader produced no samples.")

    metrics = compute_classification_metrics(
        torch.cat(all_targets).tolist(),
        torch.cat(all_predictions).tolist(),
    )

    return EpochResult(
        loss=total_loss / total_normaliser,
        unweighted_cross_entropy=total_unweighted / sample_count,
        metrics=metrics,
    )


@dataclass
class BestScoreTracker:
    """Track the best score and early-stopping patience.

    Only a strictly higher score counts as an improvement, so on ties the
    earlier epoch stays best. Training should stop once ``patience``
    consecutive epochs have not improved.
    """

    patience: int
    best_score: float = -math.inf
    best_epoch: int | None = None
    epochs_without_improvement: int = 0

    def update(self, score: float, epoch: int) -> bool:
        """Record an epoch's score and return whether it is a new best."""

        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            return True

        self.epochs_without_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_validation_macro_f1: float,
    config: dict[str, Any],
) -> None:
    """Save model and optimizer state with the context needed to use them."""

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_macro_f1": best_validation_macro_f1,
            "class_to_index": dict(CLASS_TO_INDEX),
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load a checkpoint into ``model`` (and ``optimizer``) and return it."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    if checkpoint["class_to_index"] != dict(CLASS_TO_INDEX):
        raise ValueError("Checkpoint class mapping does not match CLASS_TO_INDEX.")

    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint


def fit(
    model: nn.Module,
    train_loader: Iterable[dict[str, Any]],
    validation_loader: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_epochs: int,
    patience: int,
    run_directory: Path,
    config: dict[str, Any],
    loss_name: str = "cross_entropy",
) -> tuple[list[dict[str, Any]], BestScoreTracker]:
    """Train with validation macro-F1 model selection and early stopping.

    ``train_loss`` and ``validation_loss`` in ``history.csv`` are the
    criterion's loss (named in the ``loss_function`` column). With weighted
    cross-entropy they are not comparable to a plain cross-entropy run; the
    ``*_unweighted_cross_entropy`` columns are. Model selection always uses
    validation macro F1, never a loss.

    Writes ``history.csv`` and ``last.pt`` after every epoch, and ``best.pt``,
    ``best_validation_metrics.json``, and ``best_validation_confusion_matrix.csv``
    whenever validation macro F1 strictly improves.
    """

    tracker = BestScoreTracker(patience=patience)
    history: list[dict[str, Any]] = []

    for epoch in range(1, max_epochs + 1):
        start = time.perf_counter()
        train_result = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_result = run_epoch(model, validation_loader, criterion, device)

        score = validation_result.metrics["macro_f1"]
        improved = tracker.update(score, epoch)

        history.append(
            {
                "epoch": epoch,
                "loss_function": loss_name,
                "train_loss": train_result.loss,
                "train_unweighted_cross_entropy": train_result.unweighted_cross_entropy,
                "train_accuracy": train_result.metrics["accuracy"],
                "train_macro_f1": train_result.metrics["macro_f1"],
                "validation_loss": validation_result.loss,
                "validation_unweighted_cross_entropy": validation_result.unweighted_cross_entropy,
                "validation_accuracy": validation_result.metrics["accuracy"],
                "validation_balanced_accuracy": validation_result.metrics["balanced_accuracy"],
                "validation_macro_f1": score,
                "best_validation_macro_f1": tracker.best_score,
                "improved": improved,
                "epochs_without_improvement": tracker.epochs_without_improvement,
                "seconds": round(time.perf_counter() - start, 2),
            }
        )
        pd.DataFrame(history).to_csv(run_directory / "history.csv", index=False)

        if improved:
            save_checkpoint(
                run_directory / "best.pt", model, optimizer, epoch, tracker.best_score, config
            )
            (run_directory / "best_validation_metrics.json").write_text(
                json.dumps({"epoch": epoch, **validation_result.metrics}, indent=2)
            )
            confusion_matrix_frame(validation_result.metrics["confusion_matrix"]).to_csv(
                run_directory / "best_validation_confusion_matrix.csv"
            )

        save_checkpoint(
            run_directory / "last.pt", model, optimizer, epoch, tracker.best_score, config
        )

        unweighted_note = (
            ""
            if loss_name == "cross_entropy"
            else f" (unweighted CE {validation_result.unweighted_cross_entropy:.4f})"
        )
        print(
            f"Epoch {epoch}/{max_epochs}: "
            f"train {loss_name} {train_result.loss:.4f}, "
            f"val {loss_name} {validation_result.loss:.4f}{unweighted_note}, "
            f"val macro F1 {score:.4f}, "
            f"val balanced acc {validation_result.metrics['balanced_accuracy']:.4f}"
            f"{' (best)' if improved else ''} "
            f"[{history[-1]['seconds']:.1f}s]"
        )

        if tracker.should_stop:
            print(f"Early stopping: no improvement for {patience} epochs.")
            break

    return history, tracker
