"""Tests for HAM10000 DataLoader factories."""

import pandas as pd
import pytest
import torch

from medvision.data.dataset import HAM10000Dataset
from medvision.data.loaders import LoaderConfig, create_dataloader, create_dataloaders
from medvision.data.transforms import build_evaluation_transforms
from tests.conftest import IMAGES_PER_SPLIT, SyntheticDataset


def evaluation_dataset(synthetic_dataset: SyntheticDataset, split: str) -> HAM10000Dataset:
    return HAM10000Dataset(
        manifest_path=synthetic_dataset.manifest_path(split),
        dataset_root=synthetic_dataset.root,
        transform=build_evaluation_transforms(),
        expected_split=split,
    )


def image_id_order(loader: torch.utils.data.DataLoader) -> list[str]:
    return [image_id for batch in loader for image_id in batch["image_id"]]


def manifest_order(synthetic_dataset: SyntheticDataset, split: str) -> list[str]:
    return pd.read_csv(synthetic_dataset.manifest_path(split), dtype=str)["image_id"].tolist()


def test_batch_shape_and_types(synthetic_dataset: SyntheticDataset) -> None:
    loaders = create_dataloaders(
        synthetic_dataset.manifest_directory,
        synthetic_dataset.root,
        LoaderConfig(batch_size=4),
    )

    assert set(loaders) == {"train", "validation", "test"}

    for loader in loaders.values():
        batch = next(iter(loader))

        assert batch["image"].shape == (4, 3, 224, 224)
        assert batch["image"].dtype == torch.float32
        assert batch["target"].shape == (4,)
        assert batch["target"].dtype == torch.int64
        assert len(batch["image_id"]) == 4
        assert len(batch["lesion_id"]) == 4


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_image_and_resize_size_reach_transforms(
    synthetic_dataset: SyntheticDataset,
    split: str,
) -> None:
    loader = create_dataloaders(
        synthetic_dataset.manifest_directory,
        synthetic_dataset.root,
        LoaderConfig(batch_size=2),
        image_size=96,
        resize_size=110,
        splits=(split,),
    )[split]

    assert next(iter(loader))["image"].shape == (2, 3, 96, 96)

    if split != "train":
        resize = loader.dataset.transform.transforms[0]
        assert list(resize.size) == [110]


def test_default_config_matches_development_settings() -> None:
    config = LoaderConfig()

    assert config.batch_size == 16
    assert config.num_workers == 0
    assert config.pin_memory is False


def test_train_shuffles_by_default(synthetic_dataset: SyntheticDataset) -> None:
    loader = create_dataloader(
        evaluation_dataset(synthetic_dataset, "train"),
        "train",
        LoaderConfig(batch_size=4, seed=0),
    )

    assert image_id_order(loader) != manifest_order(synthetic_dataset, "train")
    assert sorted(image_id_order(loader)) == sorted(manifest_order(synthetic_dataset, "train"))


@pytest.mark.parametrize("split", ["validation", "test"])
def test_evaluation_order_is_manifest_order(
    synthetic_dataset: SyntheticDataset,
    split: str,
) -> None:
    loader = create_dataloader(
        evaluation_dataset(synthetic_dataset, split),
        split,
        LoaderConfig(batch_size=4, seed=0),
    )

    assert image_id_order(loader) == manifest_order(synthetic_dataset, split)
    assert image_id_order(loader) == image_id_order(loader)


def test_train_order_is_reproducible_for_same_seed(synthetic_dataset: SyntheticDataset) -> None:
    dataset = evaluation_dataset(synthetic_dataset, "train")

    first = create_dataloader(dataset, "train", LoaderConfig(batch_size=4, seed=5))
    second = create_dataloader(dataset, "train", LoaderConfig(batch_size=4, seed=5))
    other_seed = create_dataloader(dataset, "train", LoaderConfig(batch_size=4, seed=6))

    assert image_id_order(first) == image_id_order(second)
    assert image_id_order(first) != image_id_order(other_seed)


def test_persistent_workers_disabled_without_workers(synthetic_dataset: SyntheticDataset) -> None:
    loader = create_dataloader(
        evaluation_dataset(synthetic_dataset, "train"),
        "train",
        LoaderConfig(num_workers=0, persistent_workers=True),
    )

    assert loader.persistent_workers is False


def test_final_incomplete_batch_is_kept(synthetic_dataset: SyntheticDataset) -> None:
    loader = create_dataloader(
        evaluation_dataset(synthetic_dataset, "validation"),
        "validation",
        LoaderConfig(batch_size=4),
    )

    batch_sizes = [len(batch["target"]) for batch in loader]

    assert batch_sizes == [4, 4, 2]
    assert sum(batch_sizes) == IMAGES_PER_SPLIT


def test_worker_process_loading(synthetic_dataset: SyntheticDataset) -> None:
    # Starting a worker is slow on macOS (spawn), so one worker run is used.
    dataset = evaluation_dataset(synthetic_dataset, "train")
    config = LoaderConfig(batch_size=4, num_workers=1, persistent_workers=True, seed=3)
    main_process_config = LoaderConfig(batch_size=4, num_workers=0, seed=3)

    loader = create_dataloader(dataset, "train", config)

    assert loader.persistent_workers is True
    assert image_id_order(loader) == image_id_order(
        create_dataloader(dataset, "train", main_process_config)
    )


@pytest.mark.parametrize(
    "config_kwargs",
    [{"batch_size": 0}, {"num_workers": -1}],
)
def test_invalid_loader_config_fails(config_kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        LoaderConfig(**config_kwargs)


def test_unknown_split_fails(synthetic_dataset: SyntheticDataset) -> None:
    with pytest.raises(ValueError, match="Unknown split"):
        create_dataloader(evaluation_dataset(synthetic_dataset, "train"), "val")
