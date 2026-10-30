"""PyTorch dataset for HAM10000 split manifests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from medvision.data.constants import CLASS_TO_INDEX, EXPECTED_CLASSES, SPLIT_NAMES

MANIFEST_COLUMNS = (
    "image_id",
    "lesion_id",
    "class_name",
    "source",
    "relative_path",
    "split",
)


def load_manifest(
    manifest_path: Path,
    expected_split: str | None = None,
) -> pd.DataFrame:
    """Load and validate a split manifest."""

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path, dtype=str)
    validate_manifest(manifest, expected_split)

    return manifest


def validate_manifest(
    manifest: pd.DataFrame,
    expected_split: str | None = None,
) -> None:
    """Validate manifest columns, labels, uniqueness, and split."""

    missing_columns = set(MANIFEST_COLUMNS) - set(manifest.columns)

    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

    if manifest.empty:
        raise ValueError("Manifest contains no images.")

    null_counts = manifest[list(MANIFEST_COLUMNS)].isna().sum()
    null_columns = null_counts[null_counts > 0]

    if not null_columns.empty:
        raise ValueError(f"Manifest contains missing values: {null_columns.to_dict()}")

    unexpected_classes = set(manifest["class_name"]) - EXPECTED_CLASSES

    if unexpected_classes:
        raise ValueError(f"Manifest contains unknown classes: {sorted(unexpected_classes)}")

    if manifest["image_id"].duplicated().any():
        duplicated_ids = manifest.loc[manifest["image_id"].duplicated(), "image_id"]
        raise ValueError(f"Manifest contains duplicate image IDs: {duplicated_ids.tolist()[:10]}")

    if expected_split is not None:
        if expected_split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split {expected_split!r}; expected one of {SPLIT_NAMES}.")

        other_splits = set(manifest["split"]) - {expected_split}

        if other_splits:
            raise ValueError(
                f"Manifest expected to contain only {expected_split!r} but also "
                f"contains {sorted(other_splits)}."
            )


class HAM10000Dataset(Dataset[dict[str, Any]]):
    """Lazily load HAM10000 images listed in a split manifest.

    Each item is a dictionary with the transformed ``image``, the integer
    ``target`` from ``CLASS_TO_INDEX``, and the identifying metadata
    ``image_id``, ``lesion_id``, ``class_name``, and ``source``. Clinical
    metadata (age, sex, localization) is intentionally not loaded.
    """

    def __init__(
        self,
        manifest_path: Path | str,
        dataset_root: Path | str,
        transform: Callable[[Image.Image], Any] | None = None,
        expected_split: str | None = None,
        verify_files: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        self.transform = transform

        manifest = load_manifest(self.manifest_path, expected_split)
        self.records: list[dict[str, str]] = manifest[list(MANIFEST_COLUMNS)].to_dict("records")
        self.targets: list[int] = [CLASS_TO_INDEX[record["class_name"]] for record in self.records]

        # Only checks that files exist; image contents are read lazily.
        if verify_files:
            missing_paths = [
                record["relative_path"]
                for record in self.records
                if not (self.dataset_root / record["relative_path"]).is_file()
            ]

            if missing_paths:
                raise FileNotFoundError(
                    f"{len(missing_paths)} manifest images are missing under "
                    f"{self.dataset_root}, e.g. {missing_paths[:5]}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = self.dataset_root / record["relative_path"]

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # convert() returns a new in-memory image, so the file handle is closed
        # when the context manager exits.
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "target": self.targets[index],
            "image_id": record["image_id"],
            "lesion_id": record["lesion_id"],
            "class_name": record["class_name"],
            "source": record["source"],
        }
