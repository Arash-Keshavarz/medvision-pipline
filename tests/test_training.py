"""Tests for training configuration, loops, checkpoints, and the entry point."""

import json
import socket
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

import medvision.data.loaders as loaders_module
import medvision.training.baseline as baseline_module
from medvision.data.constants import CLASS_TO_INDEX, NUM_CLASSES
from medvision.data.loaders import LoaderConfig
from medvision.training.baseline import (
    SmokeSettings,
    create_run_directory,
    run_baseline,
)
from medvision.training.config import (
    DataSettings,
    TrainingConfig,
    load_data_settings,
    load_training_config,
)
from medvision.training.losses import CLASS_WEIGHT_FORMULA
from medvision.training.reproducibility import seed_everything, select_device
from medvision.training.trainer import (
    BestScoreTracker,
    fit,
    load_checkpoint,
    loss_normaliser,
    run_epoch,
    save_checkpoint,
)
from tests.conftest import SyntheticDataset, TinyModel

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("Tests must not access the network.")

    monkeypatch.setattr(socket.socket, "connect", refuse)


def make_batches(sizes: list[int], seed: int = 0) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)

    return [
        {
            "image": torch.randn(size, 3, 4, 4, generator=generator),
            "target": torch.randint(0, NUM_CLASSES, (size,), generator=generator),
        }
        for size in sizes
    ]


def parameters_of(model: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def sgd(model: nn.Module, learning_rate: float = 0.1) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=learning_rate)


# Training and evaluation loops


def test_training_epoch_updates_parameters_in_train_mode() -> None:
    torch.manual_seed(0)
    model = TinyModel().eval()
    before = parameters_of(model)

    run_epoch(model, make_batches([4, 4]), nn.CrossEntropyLoss(), CPU, sgd(model))

    assert model.modes == [True, True]
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))


def test_evaluation_does_not_update_parameters_and_uses_eval_mode() -> None:
    torch.manual_seed(0)
    model = TinyModel().train()
    before = parameters_of(model)

    run_epoch(model, make_batches([4, 4]), nn.CrossEntropyLoss(), CPU)

    assert model.modes == [False, False]
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    assert all(parameter.grad is None for parameter in model.parameters())


def test_epoch_loss_is_weighted_by_samples() -> None:
    torch.manual_seed(0)
    model = TinyModel()
    batches = make_batches([3, 1])

    result = run_epoch(model, batches, nn.CrossEntropyLoss(), CPU)

    with torch.no_grad():
        logits = [model(batch["image"]) for batch in batches]
        targets = [batch["target"] for batch in batches]
        per_sample = nn.functional.cross_entropy(torch.cat(logits), torch.cat(targets))
        batch_means = [
            nn.functional.cross_entropy(logit, target)
            for logit, target in zip(logits, targets, strict=True)
        ]

    assert result.loss == pytest.approx(per_sample.item())
    # The final batch of one sample must not get the same weight as three samples.
    assert result.loss != pytest.approx(torch.stack(batch_means).mean().item())
    assert result.metrics["sample_count"] == 4
    # Plain cross-entropy: the objective is the unweighted cross-entropy.
    assert result.unweighted_cross_entropy == pytest.approx(result.loss)


CLASS_WEIGHTS = torch.tensor([3.0, 0.5, 1.0, 7.0, 2.0, 0.25, 5.0])


def test_weighted_epoch_loss_matches_one_concatenated_calculation() -> None:
    torch.manual_seed(0)
    model = TinyModel()
    # Uneven batches with different class mixes, so batch weight sums differ.
    batches = make_batches([5, 3, 1], seed=4)
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)

    result = run_epoch(model, batches, criterion, CPU)

    with torch.no_grad():
        logits = [model(batch["image"]) for batch in batches]
        targets = [batch["target"] for batch in batches]
        all_logits, all_targets = torch.cat(logits), torch.cat(targets)
        concatenated = nn.functional.cross_entropy(all_logits, all_targets, weight=CLASS_WEIGHTS)
        unweighted = nn.functional.cross_entropy(all_logits, all_targets)
        per_sample = nn.functional.cross_entropy(
            all_logits, all_targets, weight=CLASS_WEIGHTS, reduction="none"
        )
        batch_means = [
            nn.functional.cross_entropy(logit, target, weight=CLASS_WEIGHTS)
            for logit, target in zip(logits, targets, strict=True)
        ]
        sample_weighted = sum(
            mean * len(target) for mean, target in zip(batch_means, targets, strict=True)
        ) / len(all_targets)

    assert result.loss == pytest.approx(concatenated.item())
    # PyTorch normalises by the summed target weights, not the sample count.
    assert concatenated.item() == pytest.approx(
        (per_sample.sum() / CLASS_WEIGHTS[all_targets].sum()).item()
    )
    # Neither batch-size weighting nor a plain mean of batch means is correct.
    assert result.loss != pytest.approx(sample_weighted.item())
    assert result.loss != pytest.approx(torch.stack(batch_means).mean().item())
    # The unweighted column stays comparable with a plain cross-entropy run.
    assert result.unweighted_cross_entropy == pytest.approx(unweighted.item())
    assert result.loss != pytest.approx(result.unweighted_cross_entropy)


def test_loss_normaliser() -> None:
    targets = torch.tensor([0, 3, 3, 5])

    assert loss_normaliser(nn.CrossEntropyLoss(), targets) == 4
    assert loss_normaliser(nn.CrossEntropyLoss(weight=CLASS_WEIGHTS), targets) == pytest.approx(
        3.0 + 7.0 + 7.0 + 0.25
    )

    with pytest.raises(ValueError, match="mean-reduced"):
        loss_normaliser(nn.CrossEntropyLoss(reduction="sum"), targets)

    with pytest.raises(ValueError, match="label smoothing"):
        loss_normaliser(nn.CrossEntropyLoss(weight=CLASS_WEIGHTS, label_smoothing=0.1), targets)


def test_metrics_use_all_predictions_of_the_epoch() -> None:
    torch.manual_seed(0)
    model = TinyModel()
    batches = make_batches([3, 2, 1])

    result = run_epoch(model, batches, nn.CrossEntropyLoss(), CPU)

    with torch.no_grad():
        predictions = torch.cat([model(batch["image"]).argmax(1) for batch in batches])
    targets = torch.cat([batch["target"] for batch in batches])

    assert result.metrics["accuracy"] == pytest.approx(
        (predictions == targets).float().mean().item()
    )


def test_wrong_output_shape_fails() -> None:
    model = TinyModel(num_classes=5)

    with pytest.raises(ValueError, match="Expected logits"):
        run_epoch(model, make_batches([2]), nn.CrossEntropyLoss(), CPU)


def test_non_finite_loss_fails() -> None:
    model = TinyModel()
    batches = make_batches([2])
    batches[0]["image"][0, 0, 0, 0] = float("nan")

    with pytest.raises(RuntimeError, match="Non-finite loss"):
        run_epoch(model, batches, nn.CrossEntropyLoss(), CPU, sgd(model))


def test_empty_loader_fails() -> None:
    with pytest.raises(ValueError, match="no samples"):
        run_epoch(TinyModel(), [], nn.CrossEntropyLoss(), CPU)


# Best-checkpoint selection and early stopping


def test_tracker_keeps_earlier_epoch_on_ties() -> None:
    tracker = BestScoreTracker(patience=3)

    assert tracker.update(0.5, epoch=1) is True
    assert tracker.update(0.6, epoch=2) is True
    assert tracker.update(0.6, epoch=3) is False

    assert tracker.best_epoch == 2
    assert tracker.best_score == 0.6


def test_tracker_patience_boundary() -> None:
    tracker = BestScoreTracker(patience=2)

    tracker.update(0.6, epoch=1)
    tracker.update(0.5, epoch=2)
    assert tracker.epochs_without_improvement == 1
    assert not tracker.should_stop

    tracker.update(0.6, epoch=3)
    assert tracker.epochs_without_improvement == 2
    assert tracker.should_stop

    tracker.update(0.7, epoch=4)
    assert tracker.epochs_without_improvement == 0
    assert not tracker.should_stop


def test_fit_stops_after_patience_and_keeps_first_best(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = TinyModel()
    # A zero learning rate keeps validation macro F1 constant, so every epoch
    # after the first ties with it.
    optimizer = sgd(model, learning_rate=0.0)

    history, tracker = fit(
        model,
        make_batches([4, 3]),
        make_batches([4, 2], seed=1),
        optimizer,
        nn.CrossEntropyLoss(),
        CPU,
        max_epochs=10,
        patience=2,
        run_directory=tmp_path,
        config={"test": True},
    )

    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert [row["improved"] for row in history] == [True, False, False]
    assert tracker.best_epoch == 1

    best = torch.load(tmp_path / "best.pt", weights_only=True)
    last = torch.load(tmp_path / "last.pt", weights_only=True)

    assert best["epoch"] == 1
    assert last["epoch"] == 3
    assert len(pd.read_csv(tmp_path / "history.csv")) == 3
    assert json.loads((tmp_path / "best_validation_metrics.json").read_text())["epoch"] == 1
    assert (tmp_path / "best_validation_confusion_matrix.csv").exists()


def test_fit_runs_all_epochs_while_improving(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = TinyModel()
    batches = make_batches([8, 8])

    history, _ = fit(
        model,
        batches,
        batches,
        sgd(model, learning_rate=0.5),
        nn.CrossEntropyLoss(),
        CPU,
        max_epochs=2,
        patience=5,
        run_directory=tmp_path,
        config={},
    )

    assert len(history) == 2


# Checkpoints


def test_checkpoint_round_trip_preserves_predictions(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = TinyModel()
    optimizer = sgd(model)
    run_epoch(model, make_batches([4]), nn.CrossEntropyLoss(), CPU, optimizer)
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(path, model, optimizer, epoch=3, best_validation_macro_f1=0.4, config={"a": 1})

    restored = TinyModel()
    checkpoint = load_checkpoint(path, restored, sgd(restored))
    images = make_batches([5], seed=9)[0]["image"]

    assert torch.equal(model(images), restored(images))
    assert checkpoint["epoch"] == 3
    assert checkpoint["best_validation_macro_f1"] == 0.4
    assert checkpoint["class_to_index"] == dict(CLASS_TO_INDEX)
    assert checkpoint["config"] == {"a": 1}
    assert "optimizer_state" in checkpoint


def test_checkpoint_with_different_class_mapping_is_rejected(tmp_path: Path) -> None:
    model = TinyModel()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, sgd(model), 1, 0.0, {})

    checkpoint = torch.load(path, weights_only=True)
    checkpoint["class_to_index"] = {"nv": 0}
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="class mapping"):
        load_checkpoint(path, TinyModel())


def test_run_directory_is_never_reused(tmp_path: Path) -> None:
    first = create_run_directory(tmp_path / "runs", "run-1")
    (first / "best.pt").write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        create_run_directory(tmp_path / "runs", "run-1")

    assert (first / "best.pt").read_bytes() == b"existing"


# Configuration and device selection


def test_repository_training_config_loads_with_expected_defaults() -> None:
    config = load_training_config(REPOSITORY_ROOT / "configs" / "baseline_resnet18.yaml")

    assert config == TrainingConfig(
        data_config=Path("configs/data.yaml"),
        output_root=Path("runs/baseline_resnet18"),
        seed=42,
        device="auto",
        architecture="resnet18",
        weights="imagenet",
        optimizer="adamw",
        learning_rate=1e-4,
        weight_decay=1e-4,
        loss="cross_entropy",
        max_epochs=20,
        early_stopping_patience=5,
        selection_metric="macro_f1",
    )


def test_weighted_config_differs_from_baseline_only_in_loss_and_output() -> None:
    baseline = load_training_config(REPOSITORY_ROOT / "configs" / "baseline_resnet18.yaml")
    weighted = load_training_config(REPOSITORY_ROOT / "configs" / "resnet18_weighted_ce.yaml")

    assert weighted.loss == "weighted_cross_entropy"
    assert weighted == replace(
        baseline, loss="weighted_cross_entropy", output_root=weighted.output_root
    )
    assert weighted.output_root != baseline.output_root


def test_repository_data_config_provides_training_data_settings() -> None:
    settings = load_data_settings(REPOSITORY_ROOT / "configs" / "data.yaml")

    assert settings.image_size == 224
    assert settings.resize_size == 224
    assert settings.loader == LoaderConfig(batch_size=16, num_workers=0, pin_memory=False)


@pytest.mark.parametrize(
    "overrides",
    [
        {"device": "tpu"},
        {"weights": "random"},
        {"learning_rate": 0.0},
        {"weight_decay": -1.0},
        {"max_epochs": 0},
        {"early_stopping_patience": 0},
        {"selection_metric": "accuracy"},
        {"loss": "focal"},
    ],
)
def test_invalid_training_config_values_fail(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(data_config=Path("d.yaml"), output_root=Path("runs"), **overrides)


def test_unknown_and_missing_config_keys_fail(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"

    path.write_text("data_config: d.yaml\nrun:\n  output_root: runs\n  batch_size: 8\n")
    with pytest.raises(ValueError, match="Unknown keys in 'run'"):
        load_training_config(path)

    path.write_text("run:\n  output_root: runs\n")
    with pytest.raises(ValueError, match="data_config"):
        load_training_config(path)

    with pytest.raises(FileNotFoundError):
        load_training_config(tmp_path / "missing.yaml")


def test_device_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    assert select_device("cpu") == CPU

    with pytest.raises(ValueError, match="Unknown device"):
        select_device("tpu")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="not available"):
        select_device("cuda")

    with pytest.raises(RuntimeError, match="not available"):
        select_device("mps")

    assert select_device("auto") == CPU

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert select_device("auto") == torch.device("mps")


def test_seed_everything_repeats_random_draws() -> None:
    seed_everything(7)
    first = torch.rand(3)
    seed_everything(7)

    assert torch.equal(first, torch.rand(3))


# Entry point on synthetic data


@pytest.fixture
def model_requests(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Replace ResNet18 with TinyModel in run_baseline; record ``pretrained`` per call.

    The architecture is irrelevant to these end-to-end tests, and ResNet18
    checkpoints would add about 270 MB of temporary files per run.
    tests/test_model.py covers the real ResNet18.
    """

    requests: list[bool] = []

    def build_tiny_model(pretrained: bool) -> TinyModel:
        requests.append(pretrained)
        return TinyModel()

    monkeypatch.setattr(baseline_module, "build_resnet18", build_tiny_model)
    return requests


def synthetic_settings(synthetic_dataset: SyntheticDataset) -> DataSettings:
    return DataSettings(
        dataset_root=synthetic_dataset.root,
        manifest_directory=synthetic_dataset.manifest_directory,
        image_size=64,
        resize_size=64,
        loader=LoaderConfig(batch_size=4, seed=0),
    )


def synthetic_config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(
        data_config=Path("unused.yaml"),
        output_root=tmp_path / "runs",
        device="cpu",
        weights="none",
        max_epochs=1,
    )


@pytest.mark.usefixtures("model_requests")
def test_entry_point_never_reads_the_test_split(
    synthetic_dataset: SyntheticDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened_manifests: list[str] = []
    original_dataset = loaders_module.HAM10000Dataset

    def recording_dataset(manifest_path: Path, *args: object, **kwargs: object):
        opened_manifests.append(Path(manifest_path).name)
        return original_dataset(manifest_path, *args, **kwargs)

    monkeypatch.setattr(loaders_module, "HAM10000Dataset", recording_dataset)
    synthetic_dataset.manifest_path("test").unlink()

    run_directory = run_baseline(
        synthetic_config(tmp_path), synthetic_settings(synthetic_dataset), run_name="no-test"
    )

    assert opened_manifests == ["train.csv", "validation.csv"]
    metadata = json.loads((run_directory / "run_metadata.json").read_text())
    assert set(metadata["manifests"]) == {"train", "validation"}


def test_smoke_run_writes_marked_artifacts(
    synthetic_dataset: SyntheticDataset,
    tmp_path: Path,
    model_requests: list[bool],
) -> None:
    config = replace(synthetic_config(tmp_path), weights="imagenet", max_epochs=5)

    # Smoke mode forces random weights (no download) and a single epoch.
    run_directory = run_baseline(
        config,
        synthetic_settings(synthetic_dataset),
        SmokeSettings(train_size=6, validation_size=5),
    )

    assert run_directory.name.startswith("smoke-")
    assert model_requests == [False]
    assert {path.name for path in run_directory.iterdir()} == {
        "resolved_config.json",
        "class_mapping.json",
        "run_metadata.json",
        "history.csv",
        "best.pt",
        "last.pt",
        "best_validation_metrics.json",
        "best_validation_confusion_matrix.csv",
    }

    metadata = json.loads((run_directory / "run_metadata.json").read_text())
    resolved = json.loads((run_directory / "resolved_config.json").read_text())

    assert metadata["run_type"] == "smoke"
    assert "not baseline performance" in metadata["note"]
    assert metadata["status"] == "completed"
    assert metadata["epochs_completed"] == 1
    assert metadata["manifests"]["train"]["images_used"] == 6
    assert metadata["manifests"]["validation"]["images_used"] == 5
    assert len(metadata["manifests"]["train"]["sha256"]) == 64
    assert {"commit", "dirty"} <= set(metadata["git"])
    assert resolved["training"]["weights"] == "none"
    assert json.loads((run_directory / "class_mapping.json").read_text()) == dict(CLASS_TO_INDEX)


@pytest.mark.usefixtures("model_requests")
def test_plain_run_writes_no_class_weights(
    synthetic_dataset: SyntheticDataset,
    tmp_path: Path,
) -> None:
    run_directory = run_baseline(
        synthetic_config(tmp_path), synthetic_settings(synthetic_dataset), run_name="plain"
    )

    history = pd.read_csv(run_directory / "history.csv")
    resolved = json.loads((run_directory / "resolved_config.json").read_text())
    metadata = json.loads((run_directory / "run_metadata.json").read_text())

    assert not (run_directory / "class_weights.json").exists()
    assert resolved["class_weights"] is None
    assert metadata["loss"] == "cross_entropy"
    assert metadata["note"] == "Baseline training run."
    assert (history["loss_function"] == "cross_entropy").all()
    pd.testing.assert_series_equal(
        history["validation_loss"],
        history["validation_unweighted_cross_entropy"],
        check_names=False,
    )


@pytest.mark.usefixtures("model_requests")
def test_weighted_run_saves_weight_provenance(
    synthetic_dataset: SyntheticDataset,
    tmp_path: Path,
) -> None:
    config = replace(synthetic_config(tmp_path), loss="weighted_cross_entropy")
    train_path = synthetic_dataset.manifest_path("train")
    train_classes = pd.read_csv(train_path)["class_name"]
    expected_counts = {name: int((train_classes == name).sum()) for name in CLASS_TO_INDEX}

    run_directory = run_baseline(config, synthetic_settings(synthetic_dataset), run_name="weighted")

    saved = json.loads((run_directory / "class_weights.json").read_text())
    resolved = json.loads((run_directory / "resolved_config.json").read_text())
    metadata = json.loads((run_directory / "run_metadata.json").read_text())
    checkpoint = torch.load(run_directory / "best.pt", weights_only=True)
    history = pd.read_csv(run_directory / "history.csv")

    assert saved["formula"] == CLASS_WEIGHT_FORMULA
    assert saved["class_order"] == list(CLASS_TO_INDEX)
    assert saved["counts"] == expected_counts
    assert saved["n_train"] == len(train_classes)
    assert saved["source_manifest"] == str(train_path)
    assert saved["source_manifest_sha256"] == metadata["manifests"]["train"]["sha256"]
    for name, count in expected_counts.items():
        assert saved["weights"][name] == pytest.approx(len(train_classes) / (7 * count))

    assert resolved["class_weights"] == saved
    assert checkpoint["config"]["class_weights"] == saved
    assert metadata["loss"] == "weighted_cross_entropy"
    assert "class_weights.json" in metadata["note"]
    assert (history["loss_function"] == "weighted_cross_entropy").all()
    assert history["validation_unweighted_cross_entropy"].notna().all()


@pytest.mark.usefixtures("model_requests")
def test_cpu_runs_with_same_seed_are_identical(
    synthetic_dataset: SyntheticDataset,
    tmp_path: Path,
) -> None:
    config = replace(synthetic_config(tmp_path), max_epochs=2)
    settings = synthetic_settings(synthetic_dataset)

    first = run_baseline(config, settings, run_name="first")
    second = run_baseline(config, settings, run_name="second")

    columns = ["train_loss", "validation_loss", "validation_macro_f1"]
    first_history = pd.read_csv(first / "history.csv")[columns]
    second_history = pd.read_csv(second / "history.csv")[columns]

    pd.testing.assert_frame_equal(first_history, second_history)

    first_state = torch.load(first / "last.pt", weights_only=True)["model_state"]
    second_state = torch.load(second / "last.pt", weights_only=True)["model_state"]

    assert all(torch.equal(first_state[key], second_state[key]) for key in first_state)
