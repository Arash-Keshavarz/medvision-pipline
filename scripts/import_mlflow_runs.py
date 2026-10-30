"""Import finished training run directories into local MLflow storage.

Only saved files are read: no training, inference, or data loading. Each
imported run keeps its original run name, start and end times, and source
Git commit, and is tagged ``medvision.tracking_mode=imported``. A run
directory that is already tracked in the experiment is refused.

    python scripts/import_mlflow_runs.py \\
        runs/baseline_resnet18/baseline-20261030-130625 \\
        runs/resnet18_weighted_ce/baseline-20261030-172401

After each import the MLflow run is compared with the run directory, and any
mismatch is reported as a failure.

A run directory whose live tracking FAILED is still "already tracked" in its
experiment. To record it fully without deleting the failed run, import it
into a separate experiment with --mlflow-experiment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from medvision.tracking.mlflow_tracking import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_TRACKING_DIRECTORY,
    TrackingError,
    TrackingSettings,
    import_run_directory,
    verify_run,
)
from medvision.training.baseline import git_state


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Import run directories into local MLflow.")
    parser.add_argument("run_directories", nargs="+", type=Path)
    parser.add_argument("--mlflow-dir", type=Path, default=DEFAULT_TRACKING_DIRECTORY)
    parser.add_argument("--mlflow-experiment", default=DEFAULT_EXPERIMENT_NAME)

    return parser.parse_args()


def main() -> None:
    """Import and verify each run directory; exit non-zero if any failed."""

    arguments = parse_arguments()
    settings = TrackingSettings(arguments.mlflow_dir, arguments.mlflow_experiment)
    importer_commit = git_state()["commit"]
    failures = 0

    for run_directory in arguments.run_directories:
        try:
            run_id = import_run_directory(run_directory, settings, importer_commit)
        except TrackingError as error:
            cause = f" ({error.__cause__})" if error.__cause__ else ""
            print(f"FAILED {run_directory}: {error}{cause}", file=sys.stderr)
            failures += 1
            continue

        problems = verify_run(settings, run_id, run_directory)

        if problems:
            failures += 1
            print(f"MISMATCH {run_directory} -> MLflow run {run_id}:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
        else:
            print(f"Imported and verified {run_directory} -> MLflow run {run_id}")

    print(f"Tracking URI: {settings.tracking_uri}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
