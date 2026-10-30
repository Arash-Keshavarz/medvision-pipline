"""Run the ResNet18 baseline: build loaders and model, train, save artifacts.

Only the train and validation manifests are opened; the test split is never
loaded here.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torchvision
from torch.utils.data import DataLoader, Subset

from medvision.data.constants import CLASS_TO_INDEX
from medvision.data.loaders import create_dataloader, create_dataloaders
from medvision.models.resnet import build_resnet18
from medvision.training.config import DataSettings, TrainingConfig
from medvision.training.losses import build_criterion, class_weights_from_manifest
from medvision.training.reproducibility import seed_everything, select_device
from medvision.training.trainer import fit

TRAINING_SPLITS = ("train", "validation")


@dataclass(frozen=True)
class SmokeSettings:
    """Bounded engineering check: one epoch on small deterministic subsets.

    The first ``train_size`` / ``validation_size`` manifest rows are used,
    and the model is randomly initialised, so results say nothing about
    baseline performance.
    """

    train_size: int = 128
    validation_size: int = 64

    def __post_init__(self) -> None:
        if self.train_size < 1 or self.validation_size < 1:
            raise ValueError("Smoke subset sizes must be at least 1.")


def create_run_directory(output_root: Path, run_name: str) -> Path:
    """Create a new run directory; never reuse an existing one."""

    output_root.mkdir(parents=True, exist_ok=True)
    run_directory = output_root / run_name

    try:
        run_directory.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"Run directory already exists and will not be overwritten: {run_directory}"
        ) from error

    return run_directory


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def git_state() -> dict[str, Any]:
    """Return the Git commit and whether the worktree has uncommitted changes."""

    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments], capture_output=True, text=True, check=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    status = run("status", "--porcelain")

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "note": "The commit hash does not capture uncommitted changes when dirty is true.",
    }


def environment_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "device": str(device),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.write_text(json.dumps(content, indent=2))


def build_loaders(
    data_settings: DataSettings,
    smoke: SmokeSettings | None = None,
) -> dict[str, DataLoader]:
    """Build train and validation loaders, optionally restricted for a smoke run."""

    loaders = create_dataloaders(
        manifest_directory=data_settings.manifest_directory,
        dataset_root=data_settings.dataset_root,
        config=data_settings.loader,
        image_size=data_settings.image_size,
        resize_size=data_settings.resize_size,
        splits=TRAINING_SPLITS,
    )

    if smoke is None:
        return loaders

    sizes = {"train": smoke.train_size, "validation": smoke.validation_size}

    return {
        split: create_dataloader(
            Subset(loader.dataset, range(min(sizes[split], len(loader.dataset)))),
            split,
            data_settings.loader,
        )
        for split, loader in loaders.items()
    }


def run_baseline(
    config: TrainingConfig,
    data_settings: DataSettings,
    smoke: SmokeSettings | None = None,
    run_name: str | None = None,
) -> Path:
    """Train the baseline and return the run directory."""

    if smoke is not None:
        config = replace(config, weights="none", max_epochs=1)

    start = time.perf_counter()
    started_at = datetime.now().astimezone()

    # Seed before building loaders and the model.
    seed_everything(config.seed)
    device = select_device(config.device)

    loaders = build_loaders(data_settings, smoke)
    model = build_resnet18(pretrained=config.weights == "imagenet").to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Class weights come from the full training manifest only, also in smoke
    # runs, so they never depend on validation or test labels.
    train_manifest = data_settings.manifest_directory / "train.csv"
    class_weights = (
        {
            **class_weights_from_manifest(train_manifest),
            "source_manifest_sha256": sha256_file(train_manifest),
        }
        if config.loss == "weighted_cross_entropy"
        else None
    )
    criterion = build_criterion(config.loss, class_weights).to(device)

    run_type = "smoke" if smoke is not None else "baseline"
    run_name = run_name or f"{run_type}-{started_at:%Y%m%d-%H%M%S}"
    run_directory = create_run_directory(config.output_root, run_name)

    resolved_config = {
        "run_type": run_type,
        "training": config.to_dict(),
        "data": data_settings.to_dict(),
        "smoke": None if smoke is None else vars(smoke),
        "class_weights": class_weights,
    }
    metadata = {
        "run_type": run_type,
        "note": (
            "SMOKE RUN: randomly initialised model on small subsets for one epoch. "
            "Metrics are not baseline performance."
            if smoke is not None
            else "Baseline training run."
            if config.loss == "cross_entropy"
            else "Training run with class-weighted cross-entropy (see class_weights.json)."
        ),
        "loss": config.loss,
        "status": "running",
        "started_at": started_at.isoformat(timespec="seconds"),
        "seed": config.seed,
        "loader_seed": data_settings.loader.seed,
        "environment": environment_metadata(device),
        "git": git_state(),
        "manifests": {
            split: {
                "path": str(data_settings.manifest_directory / f"{split}.csv"),
                "sha256": sha256_file(data_settings.manifest_directory / f"{split}.csv"),
                "images_used": len(loader.dataset),
            }
            for split, loader in loaders.items()
        },
    }

    _write_json(run_directory / "resolved_config.json", resolved_config)
    _write_json(run_directory / "class_mapping.json", dict(CLASS_TO_INDEX))
    _write_json(run_directory / "run_metadata.json", metadata)

    if class_weights is not None:
        _write_json(run_directory / "class_weights.json", class_weights)

    print(f"Run directory: {run_directory}")
    print(f"Device: {device} | seed: {config.seed} | weights: {config.weights}")
    print(f"Loss: {config.loss}")

    if class_weights is not None:
        print(
            "Class weights: "
            + ", ".join(f"{name} {weight:.4f}" for name, weight in class_weights["weights"].items())
        )
    print(
        f"Train images: {len(loaders['train'].dataset):,} | "
        f"validation images: {len(loaders['validation'].dataset):,}"
    )

    history, tracker = fit(
        model=model,
        train_loader=loaders["train"],
        validation_loader=loaders["validation"],
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        max_epochs=config.max_epochs,
        patience=config.early_stopping_patience,
        run_directory=run_directory,
        config=resolved_config,
        loss_name=config.loss,
    )

    metadata.update(
        {
            "status": "completed",
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "elapsed_seconds": round(time.perf_counter() - start, 1),
            "epochs_completed": len(history),
            "stopped_early": len(history) < config.max_epochs,
            "best_epoch": tracker.best_epoch,
            "best_validation_macro_f1": tracker.best_score,
        }
    )
    _write_json(run_directory / "run_metadata.json", metadata)

    return run_directory
