"""Tests for the HAM10000 PyTorch dataset."""

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from medvision.data.constants import CLASS_TO_INDEX
from medvision.data.dataset import HAM10000Dataset
from medvision.data.transforms import build_evaluation_transforms
from tests.conftest import IMAGE_SIZE, IMAGES_PER_SPLIT, SyntheticDataset, image_color


def make_dataset(
    synthetic_dataset: SyntheticDataset,
    split: str = "train",
    **kwargs: object,
) -> HAM10000Dataset:
    return HAM10000Dataset(
        manifest_path=synthetic_dataset.manifest_path(split),
        dataset_root=synthetic_dataset.root,
        **kwargs,
    )


def rewrite_manifest(synthetic_dataset: SyntheticDataset, change) -> Path:
    path = synthetic_dataset.manifest_path("train")
    manifest = pd.read_csv(path, dtype=str)
    change(manifest).to_csv(path, index=False)
    return path


def test_dataset_length(synthetic_dataset: SyntheticDataset) -> None:
    assert len(make_dataset(synthetic_dataset)) == IMAGES_PER_SPLIT


def test_item_contains_image_target_and_metadata(synthetic_dataset: SyntheticDataset) -> None:
    dataset = make_dataset(synthetic_dataset)
    manifest = pd.read_csv(synthetic_dataset.manifest_path("train"), dtype=str)

    item = dataset[3]
    row = manifest.iloc[3]

    assert set(item) == {"image", "target", "image_id", "lesion_id", "class_name", "source"}
    assert item["image_id"] == row["image_id"]
    assert item["lesion_id"] == row["lesion_id"]
    assert item["class_name"] == row["class_name"]
    assert item["source"] == row["source"]
    assert item["target"] == CLASS_TO_INDEX[row["class_name"]]


def test_image_is_loaded_correctly_as_rgb(synthetic_dataset: SyntheticDataset) -> None:
    dataset = make_dataset(synthetic_dataset)

    image = dataset[1]["image"]

    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == IMAGE_SIZE
    assert image.getpixel((0, 0)) == image_color(1)


def test_greyscale_image_is_converted_to_rgb(synthetic_dataset: SyntheticDataset) -> None:
    with Image.open(synthetic_dataset.root / "images" / "ISIC_0000000.png") as source:
        source_mode = source.mode

    image = make_dataset(synthetic_dataset)[0]["image"]

    assert source_mode == "L"
    assert image.mode == "RGB"


def test_targets_follow_canonical_mapping(synthetic_dataset: SyntheticDataset) -> None:
    dataset = make_dataset(synthetic_dataset)

    assert dataset.targets == [
        CLASS_TO_INDEX[dataset[index]["class_name"]] for index in range(len(dataset))
    ]


def test_transform_is_applied(synthetic_dataset: SyntheticDataset) -> None:
    dataset = make_dataset(synthetic_dataset, transform=build_evaluation_transforms())

    image = dataset[0]["image"]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)


@pytest.mark.parametrize("column", ["image_id", "lesion_id", "class_name", "relative_path"])
def test_missing_required_column_fails(synthetic_dataset: SyntheticDataset, column: str) -> None:
    rewrite_manifest(synthetic_dataset, lambda manifest: manifest.drop(columns=column))

    with pytest.raises(ValueError, match="missing columns"):
        make_dataset(synthetic_dataset)


def test_unknown_class_fails(synthetic_dataset: SyntheticDataset) -> None:
    def add_unknown_class(manifest: pd.DataFrame) -> pd.DataFrame:
        manifest.loc[0, "class_name"] = "melanoma"
        return manifest

    rewrite_manifest(synthetic_dataset, add_unknown_class)

    with pytest.raises(ValueError, match="unknown classes"):
        make_dataset(synthetic_dataset)


def test_duplicate_image_ids_fail(synthetic_dataset: SyntheticDataset) -> None:
    rewrite_manifest(
        synthetic_dataset,
        lambda manifest: pd.concat([manifest, manifest.iloc[[0]]], ignore_index=True),
    )

    with pytest.raises(ValueError, match="duplicate image IDs"):
        make_dataset(synthetic_dataset)


def test_expected_split_mismatch_fails(synthetic_dataset: SyntheticDataset) -> None:
    with pytest.raises(ValueError, match="expected to contain only 'validation'"):
        make_dataset(synthetic_dataset, split="train", expected_split="validation")


def test_expected_split_matches(synthetic_dataset: SyntheticDataset) -> None:
    dataset = make_dataset(synthetic_dataset, split="test", expected_split="test")

    assert len(dataset) == IMAGES_PER_SPLIT


def test_unknown_expected_split_fails(synthetic_dataset: SyntheticDataset) -> None:
    with pytest.raises(ValueError, match="Unknown split"):
        make_dataset(synthetic_dataset, expected_split="val")


def test_missing_manifest_fails(synthetic_dataset: SyntheticDataset) -> None:
    with pytest.raises(FileNotFoundError, match="Manifest file not found"):
        HAM10000Dataset(
            manifest_path=synthetic_dataset.manifest_directory / "missing.csv",
            dataset_root=synthetic_dataset.root,
        )


def test_missing_image_fails_at_construction(synthetic_dataset: SyntheticDataset) -> None:
    (synthetic_dataset.root / "images" / "ISIC_0000002.png").unlink()

    with pytest.raises(FileNotFoundError, match="1 manifest images are missing"):
        make_dataset(synthetic_dataset)


def test_missing_image_fails_on_access_without_verification(
    synthetic_dataset: SyntheticDataset,
) -> None:
    (synthetic_dataset.root / "images" / "ISIC_0000002.png").unlink()
    dataset = make_dataset(synthetic_dataset, verify_files=False)

    assert dataset[1]["image_id"] == "ISIC_0000001"

    with pytest.raises(FileNotFoundError, match="Image not found"):
        dataset[2]
