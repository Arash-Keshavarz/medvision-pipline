"""Configuration for baseline training.

Two files are combined:

- the training config (e.g. ``configs/baseline_resnet18.yaml``) owns model,
  optimisation, and run settings, and names the data config;
- the data config (``configs/data.yaml``) owns dataset paths, split
  manifests, preprocessing sizes, and DataLoader settings.

No setting is defined in both files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from medvision.data.loaders import LoaderConfig

SUPPORTED_ARCHITECTURES = ("resnet18",)
SUPPORTED_WEIGHTS = ("imagenet", "none")
SUPPORTED_OPTIMIZERS = ("adamw",)
SUPPORTED_LOSSES = ("cross_entropy", "weighted_cross_entropy")
SUPPORTED_SELECTION_METRICS = ("macro_f1",)
SUPPORTED_DEVICES = ("auto", "cpu", "cuda", "mps")


@dataclass(frozen=True)
class TrainingConfig:
    """Model, optimisation, and run settings."""

    data_config: Path
    output_root: Path
    seed: int = 42
    device: str = "auto"
    architecture: str = "resnet18"
    weights: str = "imagenet"
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    loss: str = "cross_entropy"
    max_epochs: int = 20
    early_stopping_patience: int = 5
    selection_metric: str = "macro_f1"

    def __post_init__(self) -> None:
        choices = {
            "architecture": SUPPORTED_ARCHITECTURES,
            "weights": SUPPORTED_WEIGHTS,
            "optimizer": SUPPORTED_OPTIMIZERS,
            "loss": SUPPORTED_LOSSES,
            "selection_metric": SUPPORTED_SELECTION_METRICS,
            "device": SUPPORTED_DEVICES,
        }

        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}.")

        if not self.learning_rate > 0:
            raise ValueError("learning_rate must be positive.")

        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")

        if self.max_epochs < 1:
            raise ValueError("max_epochs must be at least 1.")

        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True)
class DataSettings:
    """Data settings read from the data config."""

    dataset_root: Path
    manifest_directory: Path
    image_size: int
    resize_size: int
    loader: LoaderConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": str(self.dataset_root),
            "manifest_directory": str(self.manifest_directory),
            "image_size": self.image_size,
            "resize_size": self.resize_size,
            "loader": asdict(self.loader),
        }


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as file:
        content = yaml.safe_load(file)

    if not isinstance(content, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")

    return content


def _section(content: dict[str, Any], name: str, allowed: set[str]) -> dict[str, Any]:
    section = content.get(name, {}) or {}
    unknown = set(section) - allowed

    if unknown:
        raise ValueError(f"Unknown keys in {name!r}: {sorted(unknown)}")

    return section


def load_training_config(path: Path) -> TrainingConfig:
    """Load and validate a training config file."""

    content = _read_yaml(path)
    unknown_sections = set(content) - {"data_config", "run", "model", "optimizer", "training"}

    if unknown_sections:
        raise ValueError(f"Unknown config sections: {sorted(unknown_sections)}")

    if "data_config" not in content:
        raise ValueError("Training config must define data_config.")

    run = _section(content, "run", {"output_root", "seed", "device"})
    model = _section(content, "model", {"architecture", "weights"})
    optimizer = _section(content, "optimizer", {"name", "learning_rate", "weight_decay"})
    training = _section(
        content,
        "training",
        {"loss", "max_epochs", "early_stopping_patience", "selection_metric"},
    )

    if "output_root" not in run:
        raise ValueError("Training config must define run.output_root.")

    values: dict[str, Any] = {
        "data_config": Path(content["data_config"]),
        "output_root": Path(run["output_root"]),
        **{key: run[key] for key in ("seed", "device") if key in run},
        **model,
        **training,
    }

    if "name" in optimizer:
        values["optimizer"] = optimizer["name"]

    for key in ("learning_rate", "weight_decay"):
        if key in optimizer:
            values[key] = float(optimizer[key])

    return TrainingConfig(**values)


def load_data_settings(path: Path) -> DataSettings:
    """Load the dataset, preprocessing, and loader settings used for training."""

    content = _read_yaml(path)

    for section in ("dataset", "splits", "preprocessing", "loader"):
        if section not in content:
            raise ValueError(f"Data config is missing section {section!r}: {path}")

    return DataSettings(
        dataset_root=Path(content["dataset"]["root"]),
        manifest_directory=Path(content["splits"]["output_dir"]),
        image_size=int(content["preprocessing"]["image_size"]),
        resize_size=int(content["preprocessing"]["resize_size"]),
        loader=LoaderConfig(**content["loader"]),
    )
