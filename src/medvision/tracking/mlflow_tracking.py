"""Optional local MLflow tracking for training runs.

The run directory under ``runs/`` stays the source of truth. MLflow is an
index over it: parameters, per-epoch metrics (step = epoch), best- and
final-epoch summaries, and small files are logged; checkpoints are not
copied. Instead, ``checkpoint_reference.json`` records each checkpoint's
local path, size, and SHA-256.

Storage is always local: an SQLite database and an artifact directory
inside one tracking directory (default ``mlruns/``, ignored by Git). No
credentials or remote servers are used, and MLflow telemetry is disabled
unless the user explicitly enables it.

The same code logs a live training run and imports a finished run
directory, so both produce the same layout. Imported runs keep the original
run name, start and end times, and source Git commit; the importer's commit
is recorded separately.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from medvision.data.constants import CLASS_NAMES, CLASS_TO_INDEX
from medvision.evaluation.metrics import compute_classification_metrics

DEFAULT_TRACKING_DIRECTORY = Path("mlruns")
DEFAULT_EXPERIMENT_NAME = "ham10000-classification"

SOURCE_RUN_KEY_TAG = "medvision.source_run_key"
TRACKING_MODE_TAG = "medvision.tracking_mode"

RUN_FILES = (
    "resolved_config.json",
    "class_mapping.json",
    "run_metadata.json",
    "class_weights.json",
    "history.csv",
    "best_validation_metrics.json",
    "best_validation_confusion_matrix.csv",
)
CHECKPOINT_FILES = ("best.pt", "last.pt")
RUN_FILES_ARTIFACT_PATH = "run_files"

CHECKPOINT_NOTE = (
    "Local filesystem reference only. The checkpoint was not copied into MLflow, so "
    "this path is not a portable model artifact: it is valid only on the machine and "
    "at the path where the run was saved. The SHA-256 identifies the exact file."
)

# MLflow's per-request limits for log_batch.
_MAX_METRICS_PER_BATCH = 1000
_MAX_PARAMS_PER_BATCH = 100


class TrackingError(RuntimeError):
    """MLflow tracking could not be set up or completed."""


@dataclass(frozen=True)
class TrackingSettings:
    """Local tracking storage: ``<directory>/mlflow.db`` and ``<directory>/artifacts``."""

    directory: Path = DEFAULT_TRACKING_DIRECTORY
    experiment_name: str = DEFAULT_EXPERIMENT_NAME

    @property
    def tracking_uri(self) -> str:
        return f"sqlite:///{(self.directory / 'mlflow.db').resolve()}"

    @property
    def artifact_location(self) -> str:
        return (self.directory / "artifacts").resolve().as_uri()


def open_experiment(settings: TrackingSettings) -> tuple[Any, str]:
    """Return an MLflow client for local storage and the experiment ID."""

    # Must be set before MLflow is imported; an explicit user setting wins.
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

    try:
        from mlflow.tracking import MlflowClient
    except ImportError as error:
        raise TrackingError(
            "MLflow tracking was requested but mlflow is not installed. "
            "Install the optional extra: pip install -e '.[mlops]'"
        ) from error

    settings.directory.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=settings.tracking_uri)
    experiment = client.get_experiment_by_name(settings.experiment_name)

    if experiment is None:
        experiment_id = client.create_experiment(
            settings.experiment_name, artifact_location=settings.artifact_location
        )
    elif experiment.lifecycle_stage != "active":
        raise TrackingError(
            f"MLflow experiment {settings.experiment_name!r} is deleted; restore it or "
            "choose another experiment name."
        )
    else:
        experiment_id = experiment.experiment_id

    return client, experiment_id


# Reading run directories


@dataclass(frozen=True)
class RunDirectory:
    """Saved files of one training run directory."""

    path: Path
    resolved_config: dict[str, Any]
    metadata: dict[str, Any]
    class_weights: dict[str, Any] | None
    history: pd.DataFrame | None
    best_metrics: dict[str, Any] | None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def loss(self) -> str:
        return self.resolved_config["training"]["loss"]

    @property
    def source_run_key(self) -> str:
        """Identifies the original run: directory name plus its start time."""

        return f"{self.name}@{self.metadata['started_at']}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_history(path: Path) -> pd.DataFrame:
    """Read ``history.csv`` with exact float round-trip (pandas' default parser is not)."""

    return pd.read_csv(path, float_precision="round_trip")


def read_run_directory(path: Path) -> RunDirectory:
    """Read a run directory; history and best metrics may not exist yet."""

    path = Path(path)

    for name in ("resolved_config.json", "run_metadata.json", "class_mapping.json"):
        if not (path / name).is_file():
            raise TrackingError(f"Not a training run directory (missing {name}): {path}")

    if _read_json(path / "class_mapping.json") != dict(CLASS_TO_INDEX):
        raise TrackingError(f"Run class mapping does not match CLASS_TO_INDEX: {path}")

    resolved_config = _read_json(path / "resolved_config.json")
    weights_path = path / "class_weights.json"

    return RunDirectory(
        path=path,
        resolved_config=resolved_config,
        metadata=_read_json(path / "run_metadata.json"),
        class_weights=(
            _read_json(weights_path)
            if weights_path.is_file()
            else resolved_config.get("class_weights")
        ),
        history=read_history(path / "history.csv") if (path / "history.csv").is_file() else None,
        best_metrics=(
            _read_json(path / "best_validation_metrics.json")
            if (path / "best_validation_metrics.json").is_file()
            else None
        ),
    )


# Converting run files to MLflow tags, parameters, and metrics


def _milliseconds(timestamp: str) -> int:
    return int(datetime.fromisoformat(timestamp).timestamp() * 1000)


def _flatten(prefix: str, value: Any, out: dict[str, str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    elif isinstance(value, str):
        out[prefix] = value
    else:
        out[prefix] = json.dumps(value)


def run_tags(
    run: RunDirectory,
    tracking_mode: str,
    importer_commit: str | None = None,
) -> dict[str, str]:
    """Tags identifying the run, its source commit, and how it was tracked."""

    metadata = run.metadata
    smoke = metadata["run_type"] == "smoke"
    note = metadata.get("note", "")

    tags = {
        # MLflow's standard tag: the commit that produced the results.
        "mlflow.source.git.commit": str(metadata["git"]["commit"]),
        "mlflow.note.content": f"SMOKE RUN - NOT A RESULT. {note}" if smoke else note,
        "medvision.git_dirty": json.dumps(metadata["git"]["dirty"]),
        "medvision.source_run_id": run.name,
        "medvision.source_run_directory": str(run.path.resolve()),
        SOURCE_RUN_KEY_TAG: run.source_run_key,
        "medvision.run_type": metadata["run_type"],
        "medvision.smoke": json.dumps(smoke),
        "medvision.loss": run.loss,
        "medvision.selection_metric": "validation_macro_f1",
        TRACKING_MODE_TAG: tracking_mode,
    }

    if tracking_mode == "imported":
        tags["medvision.importer_git_commit"] = str(importer_commit)
        tags["medvision.imported_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

    return tags


def run_params(run: RunDirectory) -> dict[str, str]:
    """Configuration, seeds, provenance, environment, and class weights."""

    metadata = run.metadata
    training = run.resolved_config["training"]
    params: dict[str, str] = {
        "seed": str(metadata["seed"]),
        "loader_seed": str(metadata["loader_seed"]),
        "architecture": training["architecture"],
        "pretrained_weights": training["weights"],
        "loss": training["loss"],
    }

    config = {key: value for key, value in run.resolved_config.items() if key != "class_weights"}
    _flatten("config", config, params)
    _flatten("environment", metadata["environment"], params)
    _flatten("manifest", metadata["manifests"], params)
    _flatten("class_mapping", dict(CLASS_TO_INDEX), params)

    weights = run.class_weights

    if weights is None:
        params["class_weights"] = "none"
    else:
        params["class_weights.formula"] = weights["formula"]
        params["class_weights.n_train"] = str(weights["n_train"])
        params["class_weights.source_manifest"] = weights["source_manifest"]
        params["class_weights.source_manifest_sha256"] = str(weights.get("source_manifest_sha256"))
        for name in CLASS_NAMES:
            params[f"class_weights.count.{name}"] = str(weights["counts"][name])
            params[f"class_weights.weight.{name}"] = repr(float(weights["weights"][name]))

    return params


def epoch_metrics(row: Mapping[str, Any], loss: str) -> dict[str, float]:
    """Numeric per-epoch values from a ``history.csv`` row (step = epoch).

    Runs saved before unweighted cross-entropy was recorded used plain
    cross-entropy, so their ``*_loss`` is the unweighted cross-entropy and
    is copied to the ``*_unweighted_cross_entropy`` keys.
    """

    metrics: dict[str, float] = {}

    for key, value in row.items():
        if key in ("epoch", "loss_function"):
            continue
        metrics["epoch_seconds" if key == "seconds" else key] = float(value)

    if loss == "cross_entropy":
        for split in ("train", "validation"):
            metrics.setdefault(f"{split}_unweighted_cross_entropy", metrics[f"{split}_loss"])

    return metrics


def summary_metrics(run: RunDirectory) -> list[tuple[str, float, int]]:
    """Best-epoch (``best.*``) and final-epoch (``final.*``) values as (key, value, step)."""

    if run.history is None or run.best_metrics is None:
        raise TrackingError(f"Run has no history or best-validation metrics: {run.path}")

    history = run.history
    best = run.best_metrics
    best_epoch = int(best["epoch"])
    final_row = history.iloc[-1]
    final_epoch = int(final_row["epoch"])
    best_row = history.loc[history["epoch"] == best_epoch]

    if len(best_row) != 1:
        raise TrackingError(f"history.csv has no single row for best epoch {best_epoch}.")

    entries: list[tuple[str, float, int]] = [("best.epoch", best_epoch, best_epoch)]

    for key in ("macro_f1", "balanced_accuracy", "accuracy", "macro_precision", "macro_recall"):
        entries.append((f"best.validation_{key}", best[key], best_epoch))

    for name in CLASS_NAMES:
        for key in ("precision", "recall", "f1", "support"):
            entries.append(
                (f"best.validation_{key}.{name}", best["per_class"][name][key], best_epoch)
            )

    best_epoch_values = epoch_metrics(best_row.iloc[0].to_dict(), run.loss)
    for key in ("train_loss", "validation_loss", "validation_unweighted_cross_entropy"):
        entries.append((f"best.{key}", best_epoch_values[key], best_epoch))

    entries.append(("final.epoch", final_epoch, final_epoch))
    for key, value in epoch_metrics(final_row.to_dict(), run.loss).items():
        entries.append((f"final.{key}", value, final_epoch))

    metadata = run.metadata
    entries += [
        ("run.epochs_completed", metadata["epochs_completed"], final_epoch),
        ("run.elapsed_seconds", metadata["elapsed_seconds"], final_epoch),
        ("run.stopped_early", float(metadata["stopped_early"]), final_epoch),
    ]

    return [(key, float(value), step) for key, value, step in entries]


def checkpoint_reference(run: RunDirectory) -> dict[str, Any]:
    """Local path, size, SHA-256, and epoch of each checkpoint; nothing is copied."""

    from medvision.training.baseline import sha256_file

    epochs = {
        "best.pt": None if run.best_metrics is None else int(run.best_metrics["epoch"]),
        "last.pt": None if run.history is None else int(run.history["epoch"].iloc[-1]),
    }
    checkpoints = {}

    for name in CHECKPOINT_FILES:
        path = run.path / name
        if path.is_file():
            checkpoints[name] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "epoch": epochs[name],
            }

    return {"portable": False, "note": CHECKPOINT_NOTE, "checkpoints": checkpoints}


def _write_diagnostics(run: RunDirectory, directory: Path) -> None:
    """Small PNGs: best-epoch confusion matrix and learning curves."""

    from matplotlib.figure import Figure

    if run.best_metrics is not None:
        matrix = run.best_metrics["confusion_matrix"]
        figure = Figure(figsize=(6, 5))
        axes = figure.add_subplot()
        image = axes.imshow(matrix, cmap="Blues")
        figure.colorbar(image, ax=axes)
        axes.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        axes.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        axes.set_xlabel("Predicted")
        axes.set_ylabel("True")
        axes.set_title(f"Validation confusion matrix, best epoch {run.best_metrics['epoch']}")
        threshold = max(max(row) for row in matrix) / 2
        for i, row in enumerate(matrix):
            for j, count in enumerate(row):
                colour = "white" if count > threshold else "black"
                axes.text(j, i, str(count), ha="center", va="center", color=colour, fontsize=8)
        figure.tight_layout()
        figure.savefig(directory / "confusion_matrix_best.png", dpi=100)

    if run.history is not None:
        history = run.history
        values = [epoch_metrics(row, run.loss) for row in history.to_dict("records")]
        frame = pd.DataFrame(values, index=history["epoch"])
        figure = Figure(figsize=(10, 4))
        loss_axes, metric_axes = figure.subplots(1, 2)
        for key in ("train_unweighted_cross_entropy", "validation_unweighted_cross_entropy"):
            loss_axes.plot(frame.index, frame[key], marker="o", label=key)
        loss_axes.set_title("Unweighted cross-entropy")
        loss_axes.set_xlabel("Epoch")
        loss_axes.legend(fontsize=8)
        for key in ("validation_macro_f1", "validation_balanced_accuracy", "validation_accuracy"):
            metric_axes.plot(frame.index, frame[key], marker="o", label=key)
        if run.best_metrics is not None:
            metric_axes.axvline(run.best_metrics["epoch"], color="grey", linestyle="--")
        metric_axes.set_title("Validation metrics (dashed: best epoch)")
        metric_axes.set_xlabel("Epoch")
        metric_axes.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(directory / "learning_curves.png", dpi=100)


# Logging


def _chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _log_params(client: Any, run_id: str, params: Mapping[str, str]) -> None:
    from mlflow.entities import Param

    entries = [Param(key, value) for key, value in params.items()]
    for chunk in _chunks(entries, _MAX_PARAMS_PER_BATCH):
        client.log_batch(run_id, params=chunk)


def _log_metrics(client: Any, run_id: str, entries: list[tuple[str, float, int, int]]) -> None:
    """Log (key, value, step, timestamp_ms) entries."""

    from mlflow.entities import Metric

    metrics = [Metric(key, value, timestamp, step) for key, value, step, timestamp in entries]
    for chunk in _chunks(metrics, _MAX_METRICS_PER_BATCH):
        client.log_batch(run_id, metrics=chunk)


def _log_run_files(client: Any, run_id: str, run: RunDirectory, diagnostics: bool) -> None:
    """Log the small run files, diagnostics, and the checkpoint reference."""

    for name in RUN_FILES:
        path = run.path / name
        if path.is_file():
            client.log_artifact(run_id, str(path), RUN_FILES_ARTIFACT_PATH)

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        reference = checkpoint_reference(run)
        (directory / "checkpoint_reference.json").write_text(json.dumps(reference, indent=2))
        client.log_artifact(run_id, str(directory / "checkpoint_reference.json"), "checkpoints")

        if diagnostics:
            diagnostics_directory = directory / "diagnostics"
            diagnostics_directory.mkdir()
            _write_diagnostics(run, diagnostics_directory)
            client.log_artifacts(run_id, str(diagnostics_directory), "diagnostics")

    for name, checkpoint in reference["checkpoints"].items():
        stem = name.removesuffix(".pt")
        client.set_tag(run_id, f"medvision.checkpoint.{stem}.sha256", checkpoint["sha256"])
        client.set_tag(run_id, f"medvision.checkpoint.{stem}.local_path", checkpoint["path"])


def _log_completed_run(client: Any, run_id: str, run: RunDirectory) -> None:
    timestamp = _milliseconds(run.metadata["finished_at"])
    entries = [(key, value, step, timestamp) for key, value, step in summary_metrics(run)]
    _log_metrics(client, run_id, entries)
    _log_run_files(client, run_id, run, diagnostics=True)


def find_runs_for_source(client: Any, experiment_id: str, run: RunDirectory) -> list[str]:
    """IDs of active MLflow runs already recording this run directory."""

    found = client.search_runs(
        [experiment_id], filter_string=f"tags.`{SOURCE_RUN_KEY_TAG}` = '{run.source_run_key}'"
    )
    return [item.info.run_id for item in found]


def _start_mlflow_run(
    client: Any,
    experiment_id: str,
    run: RunDirectory,
    tags: dict[str, str],
) -> str:
    existing = find_runs_for_source(client, experiment_id, run)

    if existing:
        raise TrackingError(
            f"{run.name} is already tracked in MLflow as run(s) {existing}. Duplicate "
            "imports into one experiment are refused, even after a FAILED run, so that "
            "failed runs stay as history. To record it again without deleting anything, "
            "import it into a separate experiment (--mlflow-experiment)."
        )

    created = client.create_run(
        experiment_id,
        start_time=_milliseconds(run.metadata["started_at"]),
        tags=tags,
        run_name=run.name,
    )
    run_id = created.info.run_id
    _log_params(client, run_id, run_params(run))

    return run_id


# Importing finished runs


def import_run_directory(
    path: Path,
    settings: TrackingSettings,
    importer_commit: str | None,
) -> str:
    """Import a completed run directory without training or inference.

    Returns the new MLflow run ID. A failed import raises after attempting to
    mark the MLflow run FAILED with an ``medvision.import_error`` tag; if the
    backend itself is failing, that attempt may not succeed.
    """

    run = read_run_directory(path)

    if run.metadata.get("status") != "completed":
        raise TrackingError(
            f"Only completed runs can be imported; {run.name} has status "
            f"{run.metadata.get('status')!r}."
        )

    client, experiment_id = open_experiment(settings)
    run_id = _start_mlflow_run(
        client, experiment_id, run, run_tags(run, "imported", importer_commit)
    )

    try:
        history = run.history
        start = _milliseconds(run.metadata["started_at"])
        elapsed = history["seconds"].cumsum() * 1000
        entries = [
            (key, value, int(row["epoch"]), start + int(offset))
            for row, offset in zip(history.to_dict("records"), elapsed, strict=True)
            for key, value in epoch_metrics(row, run.loss).items()
        ]
        _log_metrics(client, run_id, entries)

        if run.loss == "cross_entropy" and "validation_unweighted_cross_entropy" not in history:
            client.set_tag(
                run_id,
                "medvision.derived_metrics",
                "*_unweighted_cross_entropy copied from *_loss (plain cross-entropy run)",
            )

        _log_completed_run(client, run_id, run)
        client.set_tag(run_id, "medvision.import_status", "complete")
        client.set_terminated(
            run_id, "FINISHED", end_time=_milliseconds(run.metadata["finished_at"])
        )
    except Exception as error:
        marked = _mark_failed(client, run_id, "medvision.import_error", error)
        state = "marked FAILED" if marked else "could not be marked FAILED; reconcile it manually"
        raise TrackingError(f"Import of {run.name} failed; MLflow run {run_id} {state}.") from error

    return run_id


def _mark_failed(client: Any, run_id: str, tag: str, error: BaseException) -> bool:
    """Try to tag the error and end the MLflow run as FAILED; return whether it worked.

    This uses the same backend that just failed, so it can fail too. The run
    may then stay RUNNING (or keep an older status) until reconciled manually.
    """

    try:
        client.set_tag(run_id, tag, f"{type(error).__name__}: {error}"[:5000])
        client.set_terminated(run_id, "FAILED")
    except Exception as cleanup_error:
        print(
            f"WARNING: could not mark MLflow run {run_id} as FAILED ({cleanup_error}); its "
            "MLflow status may be stale and need manual reconciliation.",
            file=sys.stderr,
        )
        return False

    return True


def verify_run(settings: TrackingSettings, run_id: str, path: Path) -> list[str]:
    """Compare an MLflow run with its run directory; return mismatch descriptions."""

    run = read_run_directory(path)
    client, _ = open_experiment(settings)
    tracked = client.get_run(run_id)
    problems: list[str] = []

    def check(label: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            problems.append(f"{label}: MLflow {actual!r} != saved {expected!r}")

    tags = tracked.data.tags
    check("source commit", tags.get("mlflow.source.git.commit"), run.metadata["git"]["commit"])
    check("source run key", tags.get(SOURCE_RUN_KEY_TAG), run.source_run_key)
    check("start time", tracked.info.start_time, _milliseconds(run.metadata["started_at"]))
    check("status", tracked.info.status, "FINISHED")

    for key, value in run_params(run).items():
        check(f"param {key}", tracked.data.params.get(key), value)

    # Per-epoch histories, including steps.
    expected_rows = [
        (int(row["epoch"]), epoch_metrics(row, run.loss)) for row in run.history.to_dict("records")
    ]
    for key in expected_rows[0][1]:
        history = sorted(client.get_metric_history(run_id, key), key=lambda metric: metric.step)
        check(
            f"history {key}",
            [(metric.step, metric.value) for metric in history],
            [(epoch, values[key]) for epoch, values in expected_rows],
        )

    latest = tracked.data.metrics
    for key, value, _ in summary_metrics(run):
        check(f"metric {key}", latest.get(key), value)

    # The saved metrics must follow from the saved confusion matrix.
    matrix = pd.read_csv(path / "best_validation_confusion_matrix.csv", index_col=0)
    targets, predictions = [], []
    for true_index, row in enumerate(matrix.to_numpy()):
        for predicted_index, count in enumerate(row):
            targets += [true_index] * int(count)
            predictions += [predicted_index] * int(count)
    recomputed = compute_classification_metrics(targets, predictions)
    for key in ("macro_f1", "balanced_accuracy", "accuracy"):
        if abs(recomputed[key] - latest.get(f"best.validation_{key}", float("nan"))) > 1e-12:
            problems.append(f"best.validation_{key} does not match the confusion matrix")

    with tempfile.TemporaryDirectory() as temporary:
        for name in RUN_FILES:
            if (path / name).is_file():
                downloaded = client.download_artifacts(
                    run_id, f"{RUN_FILES_ARTIFACT_PATH}/{name}", temporary
                )
                if Path(downloaded).read_bytes() != (path / name).read_bytes():
                    problems.append(f"artifact {name} differs from the run directory")

    logged = {item.path for item in client.list_artifacts(run_id, "checkpoints")}
    if any(item.endswith(".pt") for item in logged):
        problems.append("checkpoint files were copied into MLflow")

    return problems


# Live tracking during training


class MlflowRunTracker:
    """Track one live training run.

    ``start`` raises if tracking cannot be set up, so a long run never
    starts without the tracking that was requested. Failures after that do
    not stop training (the run directory is complete either way): they are
    printed and kept in ``error``, and marking the MLflow run FAILED is
    attempted.
    """

    def __init__(self, settings: TrackingSettings | None = None) -> None:
        self.settings = settings or TrackingSettings()
        self.client: Any = None
        self.run_id: str | None = None
        self.error: str | None = None

    def start(self, run_directory: Path) -> dict[str, Any]:
        """Create the MLflow run and log parameters; return a summary for run metadata."""

        try:
            self.client, experiment_id = open_experiment(self.settings)
            run = read_run_directory(run_directory)
            self.run_id = _start_mlflow_run(self.client, experiment_id, run, run_tags(run, "live"))
        except Exception as error:
            if self.run_id is not None:
                _mark_failed(self.client, self.run_id, "medvision.tracking_error", error)
            raise TrackingError(f"Could not start MLflow tracking: {error}") from error

        return {
            "backend": "mlflow",
            "tracking_uri": self.settings.tracking_uri,
            "experiment": self.settings.experiment_name,
            "run_id": self.run_id,
        }

    def _fail(self, action: str, error: BaseException) -> None:
        self.error = f"{action}: {type(error).__name__}: {error}"
        print(
            f"WARNING: MLflow tracking failed ({self.error}). Training continues and the run "
            f"directory stays complete. MLflow run {self.run_id} is kept (marking it FAILED "
            "is attempted); to record the run fully, import the run directory into a "
            "separate experiment with scripts/import_mlflow_runs.py --mlflow-experiment <name>.",
            file=sys.stderr,
        )
        _mark_failed(self.client, self.run_id, "medvision.tracking_error", error)

    def log_epoch(self, row: Mapping[str, Any]) -> None:
        """Log one ``history.csv`` row with step = epoch."""

        if self.error is not None:
            return

        try:
            timestamp = int(time.time() * 1000)
            loss = str(row.get("loss_function", "cross_entropy"))
            entries = [
                (key, value, int(row["epoch"]), timestamp)
                for key, value in epoch_metrics(row, loss).items()
            ]
            _log_metrics(self.client, self.run_id, entries)
        except Exception as error:
            self._fail(f"logging epoch {row.get('epoch')}", error)

    def finish(self, run_directory: Path) -> None:
        """Log best/final summaries and files, then mark the run FINISHED."""

        if self.error is not None:
            return

        try:
            run = read_run_directory(run_directory)
            _log_completed_run(self.client, self.run_id, run)
            self.client.set_terminated(
                self.run_id, "FINISHED", end_time=_milliseconds(run.metadata["finished_at"])
            )
        except Exception as error:
            self._fail("finishing the run", error)

    def abort(self, run_directory: Path, status: str) -> None:
        """Log whatever files exist and end the run as FAILED or KILLED; never raises.

        If tracking already failed, nothing more is sent to MLflow.
        """

        if self.error is not None:
            return

        try:
            run = read_run_directory(run_directory)
            _log_run_files(self.client, self.run_id, run, diagnostics=False)
            self.client.set_tag(self.run_id, "medvision.run_status", run.metadata.get("status"))
            self.client.set_terminated(self.run_id, status)
        except Exception as error:
            self._fail(f"closing the run as {status}", error)
