"""Tests for local MLflow tracking and run import (synthetic data, no network)."""

import json
import socket
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch

import medvision.training.baseline as baseline_module
from medvision.data.constants import CLASS_NAMES
from medvision.data.loaders import LoaderConfig
from medvision.tracking.mlflow_tracking import (
    CHECKPOINT_NOTE,
    MlflowRunTracker,
    TrackingError,
    TrackingSettings,
    epoch_metrics,
    import_run_directory,
    open_experiment,
    read_history,
    read_run_directory,
    verify_run,
)
from medvision.training.baseline import SmokeSettings, run_baseline, sha256_file
from medvision.training.config import DataSettings, TrainingConfig
from tests.conftest import SyntheticDataset, TinyModel

pytest.importorskip("mlflow")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "a" * 40
IMPORTER_COMMIT = "b" * 40


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("Tests must not access the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture(autouse=True)
def tiny_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # ResNet18 checkpoints would add ~270 MB per run; the architecture is irrelevant here.
    monkeypatch.setattr(baseline_module, "build_resnet18", lambda pretrained: TinyModel())


@pytest.fixture(scope="session")
def tracking_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # One database for the session: creating its schema takes seconds.
    return tmp_path_factory.mktemp("mlruns")


@pytest.fixture
def settings(tracking_directory: Path, request: pytest.FixtureRequest) -> TrackingSettings:
    # A separate experiment per test keeps tests independent.
    return TrackingSettings(tracking_directory, experiment_name=request.node.name)


@pytest.fixture
def fixed_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        baseline_module,
        "git_state",
        lambda: {"commit": SOURCE_COMMIT, "dirty": False, "note": "test"},
    )


def data_settings(synthetic_dataset: SyntheticDataset) -> DataSettings:
    return DataSettings(
        dataset_root=synthetic_dataset.root,
        manifest_directory=synthetic_dataset.manifest_directory,
        image_size=64,
        resize_size=64,
        loader=LoaderConfig(batch_size=4, seed=0),
    )


def training_config(tmp_path: Path, **overrides: object) -> TrainingConfig:
    values = {
        "data_config": Path("unused.yaml"),
        "output_root": tmp_path / "runs",
        "device": "cpu",
        "weights": "none",
        "max_epochs": 2,
        **overrides,
    }
    return TrainingConfig(**values)


def train(
    synthetic_dataset: SyntheticDataset,
    tmp_path: Path,
    tracker: MlflowRunTracker | None = None,
    run_name: str = "run",
    **overrides: object,
) -> Path:
    return run_baseline(
        training_config(tmp_path, **overrides),
        data_settings(synthetic_dataset),
        run_name=run_name,
        tracker=tracker,
    )


def get_run(settings: TrackingSettings, run_id: str):
    client, _ = open_experiment(settings)
    return client, client.get_run(run_id)


def logged_metadata_bytes(settings: TrackingSettings, run_id: str, destination: Path) -> bytes:
    client, _ = open_experiment(settings)
    destination.mkdir(exist_ok=True)
    path = client.download_artifacts(run_id, "run_files/run_metadata.json", str(destination))
    return Path(path).read_bytes()


# Live tracking


@pytest.mark.usefixtures("fixed_commit")
def test_live_run_records_provenance_epochs_and_summaries(
    synthetic_dataset: SyntheticDataset, tmp_path: Path, settings: TrackingSettings
) -> None:
    tracker = MlflowRunTracker(settings)
    run_directory = train(
        synthetic_dataset, tmp_path, tracker, loss="weighted_cross_entropy", max_epochs=3
    )

    metadata = json.loads((run_directory / "run_metadata.json").read_text())
    history = read_history(run_directory / "history.csv")
    best = json.loads((run_directory / "best_validation_metrics.json").read_text())
    client, run = get_run(settings, tracker.run_id)
    params, tags, metrics = run.data.params, run.data.tags, run.data.metrics

    assert run.info.status == "FINISHED"
    assert run.info.run_name == run_directory.name
    assert metadata["tracking"]["run_id"] == tracker.run_id
    assert metadata["tracking"]["mlflow_status"] == "FINISHED"
    assert metadata["tracking"]["error"] is None

    # Provenance.
    assert tags["mlflow.source.git.commit"] == SOURCE_COMMIT
    assert tags["medvision.git_dirty"] == "false"
    assert tags["medvision.tracking_mode"] == "live"
    assert tags["medvision.smoke"] == "false"
    assert params["seed"] == "42"
    assert params["loader_seed"] == "0"
    assert params["architecture"] == "resnet18"
    assert params["pretrained_weights"] == "none"
    assert params["loss"] == "weighted_cross_entropy"
    assert params["manifest.train.sha256"] == metadata["manifests"]["train"]["sha256"]
    assert params["manifest.validation.sha256"] == metadata["manifests"]["validation"]["sha256"]
    assert params["environment.device"] == "cpu"
    assert params["environment.torch"] == torch.__version__
    assert params["class_mapping.mel"] == "4"
    assert "class_weights.formula" in params
    for name in CLASS_NAMES:
        assert f"class_weights.weight.{name}" in params

    # Per-epoch metrics with explicit steps.
    macro_f1 = client.get_metric_history(tracker.run_id, "validation_macro_f1")
    assert sorted((metric.step, metric.value) for metric in macro_f1) == list(
        zip(history["epoch"], history["validation_macro_f1"], strict=True)
    )

    # Best epoch and final epoch are separate.
    assert metrics["best.epoch"] == best["epoch"]
    assert metrics["best.validation_macro_f1"] == best["macro_f1"]
    assert metrics["best.validation_recall.mel"] == best["per_class"]["mel"]["recall"]
    assert metrics["final.epoch"] == history["epoch"].iloc[-1]
    assert metrics["final.validation_macro_f1"] == history["validation_macro_f1"].iloc[-1]

    assert verify_run(settings, tracker.run_id, run_directory) == []


@pytest.mark.usefixtures("fixed_commit")
def test_logged_metadata_matches_final_local_metadata(
    synthetic_dataset: SyntheticDataset, tmp_path: Path, settings: TrackingSettings
) -> None:
    tracker = MlflowRunTracker(settings)
    run_directory = train(synthetic_dataset, tmp_path, tracker)

    local = (run_directory / "run_metadata.json").read_bytes()
    logged = logged_metadata_bytes(settings, tracker.run_id, tmp_path / "download")
    metadata = json.loads(logged)

    # The MLflow copy is the final local file, not an earlier version.
    assert logged == local
    assert metadata["status"] == "completed"
    assert metadata["tracking"]["mlflow_status"] == "FINISHED"
    assert metadata["tracking"]["error"] is None


@pytest.mark.usefixtures("fixed_commit")
def test_checkpoints_are_referenced_not_copied(
    synthetic_dataset: SyntheticDataset, tmp_path: Path, settings: TrackingSettings
) -> None:
    tracker = MlflowRunTracker(settings)
    run_directory = train(synthetic_dataset, tmp_path, tracker)
    client, run = get_run(settings, tracker.run_id)

    artifact_paths = {item.path for item in client.list_artifacts(tracker.run_id)}
    all_paths = {
        item.path
        for folder in artifact_paths
        for item in client.list_artifacts(tracker.run_id, folder)
    }
    assert not any(path.endswith(".pt") for path in all_paths)
    assert "diagnostics/confusion_matrix_best.png" in all_paths
    assert "diagnostics/learning_curves.png" in all_paths
    assert "run_files/best_validation_confusion_matrix.csv" in all_paths

    reference_path = client.download_artifacts(
        tracker.run_id, "checkpoints/checkpoint_reference.json", str(tmp_path)
    )
    reference = json.loads(Path(reference_path).read_text())

    assert reference["portable"] is False
    assert reference["note"] == CHECKPOINT_NOTE
    best = reference["checkpoints"]["best.pt"]
    assert best["path"] == str((run_directory / "best.pt").resolve())
    assert best["sha256"] == sha256_file(run_directory / "best.pt")
    assert best["size_bytes"] == (run_directory / "best.pt").stat().st_size
    assert run.data.tags["medvision.checkpoint.best.sha256"] == best["sha256"]


@pytest.mark.usefixtures("fixed_commit")
def test_smoke_runs_are_tagged(
    synthetic_dataset: SyntheticDataset, tmp_path: Path, settings: TrackingSettings
) -> None:
    tracker = MlflowRunTracker(settings)
    run_baseline(
        training_config(tmp_path),
        data_settings(synthetic_dataset),
        SmokeSettings(train_size=6, validation_size=5),
        tracker=tracker,
    )
    _, run = get_run(settings, tracker.run_id)

    assert run.data.tags["medvision.smoke"] == "true"
    assert run.data.tags["medvision.run_type"] == "smoke"
    assert run.data.tags["mlflow.note.content"].startswith("SMOKE RUN - NOT A RESULT.")
    assert run.info.run_name.startswith("smoke-")


@pytest.mark.usefixtures("fixed_commit")
def test_tracking_does_not_change_local_artifacts(
    synthetic_dataset: SyntheticDataset, tmp_path: Path, settings: TrackingSettings
) -> None:
    plain = train(synthetic_dataset, tmp_path, run_name="untracked")
    tracked = train(synthetic_dataset, tmp_path, MlflowRunTracker(settings), run_name="tracked")

    assert {path.name for path in plain.iterdir()} == {path.name for path in tracked.iterdir()}
    pd.testing.assert_frame_equal(
        pd.read_csv(plain / "history.csv").drop(columns="seconds"),
        pd.read_csv(tracked / "history.csv").drop(columns="seconds"),
    )
    assert "tracking" not in json.loads((plain / "run_metadata.json").read_text())


@pytest.mark.usefixtures("fixed_commit")
@pytest.mark.parametrize(
    ("error", "local_status", "mlflow_status"),
    [(RuntimeError("boom"), "failed", "FAILED"), (KeyboardInterrupt(), "interrupted", "KILLED")],
)
def test_failed_and_interrupted_runs_are_closed(
    synthetic_dataset: SyntheticDataset,
    tmp_path: Path,
    settings: TrackingSettings,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    local_status: str,
    mlflow_status: str,
) -> None:
    original_fit = baseline_module.fit

    def fit_then_fail(*args: object, **kwargs: object):
        callback = kwargs["epoch_callback"]

        def callback_then_fail(row: dict) -> None:
            callback(row)
            raise error

        return original_fit(*args, **{**kwargs, "epoch_callback": callback_then_fail})

    monkeypatch.setattr(baseline_module, "fit", fit_then_fail)
    tracker = MlflowRunTracker(settings)

    with pytest.raises(type(error)):
        train(synthetic_dataset, tmp_path, tracker)

    metadata = json.loads((tmp_path / "runs" / "run" / "run_metadata.json").read_text())
    client, run = get_run(settings, tracker.run_id)

    assert metadata["status"] == local_status
    assert metadata["tracking"]["mlflow_status"] == mlflow_status
    assert run.info.status == mlflow_status
    logged = logged_metadata_bytes(settings, tracker.run_id, tmp_path / "download")
    assert logged == (tmp_path / "runs" / "run" / "run_metadata.json").read_bytes()
    assert run.data.tags["medvision.run_status"] == local_status
    # The completed epoch was logged before the failure.
    assert [metric.step for metric in client.get_metric_history(tracker.run_id, "train_loss")] == [
        1
    ]


@pytest.mark.usefixtures("fixed_commit")
def test_tracking_failure_is_visible_and_training_completes(
    synthetic_dataset: SyntheticDataset,
    tmp_path: Path,
    settings: TrackingSettings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tracker = MlflowRunTracker(settings)

    def broken_log_epoch(row: dict) -> None:
        tracker._fail("logging epoch", OSError("disk full"))

    monkeypatch.setattr(tracker, "log_epoch", broken_log_epoch)
    run_directory = train(synthetic_dataset, tmp_path, tracker)

    metadata = json.loads((run_directory / "run_metadata.json").read_text())
    _, run = get_run(settings, tracker.run_id)

    assert metadata["status"] == "completed"
    assert metadata["tracking"]["mlflow_status"] == "FAILED"
    assert "disk full" in metadata["tracking"]["error"]
    assert run.info.status == "FAILED"
    assert "disk full" in run.data.tags["medvision.tracking_error"]
    assert "WARNING: MLflow tracking failed" in capsys.readouterr().err


@pytest.mark.usefixtures("fixed_commit")
def test_failed_run_is_kept_and_recovered_in_a_separate_experiment(
    synthetic_dataset: SyntheticDataset,
    tmp_path: Path,
    settings: TrackingSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = MlflowRunTracker(settings)
    monkeypatch.setattr(
        tracker, "log_epoch", lambda row: tracker._fail("logging epoch", OSError("disk full"))
    )
    run_directory = train(synthetic_dataset, tmp_path, tracker)

    # Re-importing into the same experiment is refused; the FAILED run stays.
    with pytest.raises(TrackingError, match="separate experiment"):
        import_run_directory(run_directory, settings, IMPORTER_COMMIT)

    recovery = replace(settings, experiment_name=f"{settings.experiment_name}-recovered")
    run_id = import_run_directory(run_directory, recovery, IMPORTER_COMMIT)

    assert verify_run(recovery, run_id, run_directory) == []
    _, failed = get_run(settings, tracker.run_id)
    assert failed.info.status == "FAILED"
    assert failed.info.lifecycle_stage == "active"


def test_tracking_start_failure_stops_before_training(
    synthetic_dataset: SyntheticDataset,
    tmp_path: Path,
    settings: TrackingSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_training(*args: object, **kwargs: object) -> None:
        raise AssertionError("Training must not start without the requested tracking.")

    monkeypatch.setattr(baseline_module, "fit", no_training)
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    tracker = MlflowRunTracker(replace(settings, directory=blocker))

    with pytest.raises(TrackingError, match="Could not start MLflow tracking"):
        train(synthetic_dataset, tmp_path, tracker)

    metadata = json.loads((tmp_path / "runs" / "run" / "run_metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert "TrackingError" in metadata["error"]


# Import


@pytest.fixture
def completed_run(
    synthetic_dataset: SyntheticDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.setattr(
        baseline_module,
        "git_state",
        lambda: {"commit": SOURCE_COMMIT, "dirty": False, "note": "test"},
    )
    return train(synthetic_dataset, tmp_path, run_name="baseline-20260101-000000", max_epochs=3)


def test_import_preserves_source_identity_and_metrics(
    completed_run: Path, settings: TrackingSettings
) -> None:
    metadata = json.loads((completed_run / "run_metadata.json").read_text())

    run_id = import_run_directory(completed_run, settings, IMPORTER_COMMIT)
    client, run = get_run(settings, run_id)
    tags = run.data.tags

    assert run.info.status == "FINISHED"
    assert run.info.run_name == "baseline-20260101-000000"
    assert tags["medvision.source_run_id"] == "baseline-20260101-000000"
    assert tags["medvision.tracking_mode"] == "imported"
    assert tags["medvision.import_status"] == "complete"
    # Metrics belong to the commit that produced them, not the importer.
    assert tags["mlflow.source.git.commit"] == SOURCE_COMMIT
    assert tags["medvision.importer_git_commit"] == IMPORTER_COMMIT

    started = pd.Timestamp(metadata["started_at"]).timestamp() * 1000
    finished = pd.Timestamp(metadata["finished_at"]).timestamp() * 1000
    assert run.info.start_time == int(started)
    assert run.info.end_time == int(finished)

    steps = [metric.step for metric in client.get_metric_history(run_id, "validation_loss")]
    assert sorted(steps) == [1, 2, 3]
    assert verify_run(settings, run_id, completed_run) == []


def test_duplicate_import_is_refused(completed_run: Path, settings: TrackingSettings) -> None:
    run_id = import_run_directory(completed_run, settings, IMPORTER_COMMIT)

    with pytest.raises(TrackingError, match=f"already tracked.*{run_id}"):
        import_run_directory(completed_run, settings, IMPORTER_COMMIT)

    client, experiment_id = open_experiment(settings)
    assert len(client.search_runs([experiment_id])) == 1


def test_live_tracked_run_cannot_be_imported_again(
    synthetic_dataset: SyntheticDataset,
    tmp_path: Path,
    settings: TrackingSettings,
    fixed_commit: None,
) -> None:
    tracker = MlflowRunTracker(settings)
    run_directory = train(synthetic_dataset, tmp_path, tracker)

    with pytest.raises(TrackingError, match="already tracked"):
        import_run_directory(run_directory, settings, IMPORTER_COMMIT)


def test_incomplete_run_is_not_imported(completed_run: Path, settings: TrackingSettings) -> None:
    path = completed_run / "run_metadata.json"
    metadata = json.loads(path.read_text())
    path.write_text(json.dumps({**metadata, "status": "running"}))

    with pytest.raises(TrackingError, match="Only completed runs"):
        import_run_directory(completed_run, settings, IMPORTER_COMMIT)


def test_import_of_older_plain_run_derives_unweighted_cross_entropy(
    completed_run: Path, settings: TrackingSettings
) -> None:
    # Runs saved before Milestone 4's loss columns had no unweighted CE columns.
    history_path = completed_run / "history.csv"
    old = pd.read_csv(history_path).drop(
        columns=[
            "loss_function",
            "train_unweighted_cross_entropy",
            "validation_unweighted_cross_entropy",
        ]
    )
    old.to_csv(history_path, index=False)

    run_id = import_run_directory(completed_run, settings, IMPORTER_COMMIT)
    client, run = get_run(settings, run_id)

    history = client.get_metric_history(run_id, "validation_unweighted_cross_entropy")
    assert sorted(metric.value for metric in history) == sorted(old["validation_loss"])
    assert "copied from *_loss" in run.data.tags["medvision.derived_metrics"]
    assert verify_run(settings, run_id, completed_run) == []


def test_verification_detects_changed_artifacts(
    completed_run: Path, settings: TrackingSettings
) -> None:
    run_id = import_run_directory(completed_run, settings, IMPORTER_COMMIT)

    history_path = completed_run / "history.csv"
    history = pd.read_csv(history_path)
    history.loc[0, "validation_macro_f1"] = 0.123
    history.to_csv(history_path, index=False)

    problems = verify_run(settings, run_id, completed_run)
    assert any("history validation_macro_f1" in problem for problem in problems)
    assert any("artifact history.csv" in problem for problem in problems)


def test_failed_import_is_marked_and_raised(
    completed_run: Path, settings: TrackingSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import medvision.tracking.mlflow_tracking as tracking_module

    def broken(*args: object, **kwargs: object) -> None:
        raise OSError("artifact store unavailable")

    monkeypatch.setattr(tracking_module, "_log_completed_run", broken)

    with pytest.raises(TrackingError, match="marked FAILED"):
        import_run_directory(completed_run, settings, IMPORTER_COMMIT)

    client, experiment_id = open_experiment(settings)
    (run,) = client.search_runs([experiment_id])
    assert run.info.status == "FAILED"
    assert "artifact store unavailable" in run.data.tags["medvision.import_error"]


def test_non_run_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(TrackingError, match="Not a training run directory"):
        read_run_directory(tmp_path)


def test_epoch_metrics_keep_numeric_values_only() -> None:
    row = {
        "epoch": 2,
        "loss_function": "weighted_cross_entropy",
        "train_loss": 0.5,
        "validation_loss": 0.7,
        "improved": True,
        "seconds": 3.5,
    }

    assert epoch_metrics(row, "weighted_cross_entropy") == {
        "train_loss": 0.5,
        "validation_loss": 0.7,
        "improved": 1.0,
        "epoch_seconds": 3.5,
    }


def test_default_tracking_storage_is_local_and_ignored_by_git() -> None:
    settings = TrackingSettings()

    assert settings.tracking_uri.startswith("sqlite:///")
    assert settings.tracking_uri.endswith("mlruns/mlflow.db")
    assert settings.artifact_location.startswith("file://")
    assert "mlruns/" in (REPOSITORY_ROOT / ".gitignore").read_text().splitlines()


@pytest.mark.usefixtures("fixed_commit")
def test_close_failure_is_recorded_locally_when_backend_keeps_failing(
    synthetic_dataset: SyntheticDataset,
    tmp_path: Path,
    settings: TrackingSettings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tracker = MlflowRunTracker(settings)
    original_start = tracker.start

    def start_then_break_backend(run_directory: Path) -> dict:
        summary = original_start(run_directory)

        def unavailable(*args: object, **kwargs: object) -> None:
            raise OSError("backend unavailable")

        # Epochs log normally; every later write, including marking FAILED, fails.
        for name in ("log_artifact", "log_artifacts", "set_tag", "set_terminated"):
            monkeypatch.setattr(tracker.client, name, unavailable)
        return summary

    monkeypatch.setattr(tracker, "start", start_then_break_backend)
    run_directory = train(synthetic_dataset, tmp_path, tracker)

    metadata = json.loads((run_directory / "run_metadata.json").read_text())
    _, run = get_run(settings, tracker.run_id)

    # The local file records the failure; the MLflow status is not guaranteed.
    assert metadata["status"] == "completed"
    assert metadata["tracking"]["mlflow_status"] == "FAILED"
    assert "backend unavailable" in metadata["tracking"]["error"]
    assert run.info.status == "RUNNING"
    assert "manual reconciliation" in capsys.readouterr().err
