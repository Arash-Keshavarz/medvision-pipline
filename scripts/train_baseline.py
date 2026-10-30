"""Train the ResNet18 HAM10000 baseline on the train and validation splits.

Normal run (pretrained ImageNet weights, settings from the config):

    python scripts/train_baseline.py --config configs/baseline_resnet18.yaml

Offline smoke run (random initialisation, one epoch, small subsets):

    python scripts/train_baseline.py --smoke

Optional local MLflow tracking (stored under mlruns/, ignored by Git):

    python scripts/train_baseline.py --config configs/baseline_resnet18.yaml --mlflow

The test split is never loaded by this script.
"""

from __future__ import annotations

import argparse
import signal
import sys
from dataclasses import replace
from pathlib import Path

from medvision.tracking.mlflow_tracking import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_TRACKING_DIRECTORY,
    MlflowRunTracker,
    TrackingSettings,
)
from medvision.training.baseline import SmokeSettings, run_baseline
from medvision.training.config import load_data_settings, load_training_config

DEFAULT_CONFIG_PATH = Path("configs/baseline_resnet18.yaml")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train the ResNet18 HAM10000 baseline (train/validation only)."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Training config; it names the data config.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=None,
        help="Override run.device from the config.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run directory name (default: <run type>-<timestamp>). Must not exist.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Engineering check: random weights (no download), one epoch, and the first "
            "--smoke-train-size / --smoke-validation-size manifest rows."
        ),
    )
    parser.add_argument("--smoke-train-size", type=int, default=SmokeSettings.train_size)
    parser.add_argument("--smoke-validation-size", type=int, default=SmokeSettings.validation_size)
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Also track the run in local MLflow storage (off by default).",
    )
    parser.add_argument(
        "--mlflow-dir",
        type=Path,
        default=DEFAULT_TRACKING_DIRECTORY,
        help="Local MLflow directory (SQLite database and artifacts).",
    )
    parser.add_argument("--mlflow-experiment", default=DEFAULT_EXPERIMENT_NAME)

    return parser.parse_args()


def raise_keyboard_interrupt(signal_number: int, frame: object) -> None:
    """Treat SIGTERM like Ctrl-C so the run is closed as interrupted."""

    raise KeyboardInterrupt(f"signal {signal_number}")


def main() -> None:
    """Load the configs and run training."""

    arguments = parse_arguments()

    config = load_training_config(arguments.config)

    if arguments.device is not None:
        config = replace(config, device=arguments.device)

    data_settings = load_data_settings(config.data_config)
    smoke = (
        SmokeSettings(arguments.smoke_train_size, arguments.smoke_validation_size)
        if arguments.smoke
        else None
    )

    tracker = (
        MlflowRunTracker(TrackingSettings(arguments.mlflow_dir, arguments.mlflow_experiment))
        if arguments.mlflow
        else None
    )
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)

    run_directory = run_baseline(config, data_settings, smoke, arguments.run_name, tracker)

    print(f"Artifacts saved to: {run_directory}")

    if tracker is not None and tracker.error is not None:
        print(f"ERROR: MLflow tracking failed: {tracker.error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
