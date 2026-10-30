"""Shared synthetic HAM10000 fixtures for preprocessing tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from medvision.data.constants import CLASS_NAMES, SPLIT_NAMES

IMAGES_PER_SPLIT = 10
IMAGE_SIZE = (60, 45)


@dataclass(frozen=True)
class SyntheticDataset:
    root: Path
    manifest_directory: Path

    def manifest_path(self, split: str) -> Path:
        return self.manifest_directory / f"{split}.csv"


def image_color(index: int) -> tuple[int, int, int]:
    """Return a distinct, deterministic colour for an image index."""

    return (20 * (index % 12), 255 - 15 * (index % 16), 7 * (index % 36))


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> SyntheticDataset:
    """Create small PNG images and one manifest per split.

    The first image of each split is saved in greyscale ("L") mode to exercise
    RGB conversion. PNG keeps pixel values exact.
    """

    root = tmp_path / "ham10000"
    manifest_directory = tmp_path / "splits"
    (root / "images").mkdir(parents=True)

    image_index = 0

    for split in SPLIT_NAMES:
        rows = []

        for position in range(IMAGES_PER_SPLIT):
            image_id = f"ISIC_{image_index:07d}"
            relative_path = f"images/{image_id}.png"
            image = Image.new("RGB", IMAGE_SIZE, image_color(image_index))

            if position == 0:
                image = image.convert("L")

            image.save(root / relative_path)

            rows.append(
                {
                    "image_id": image_id,
                    "lesion_id": f"HAM_{image_index // 2:07d}_{split}",
                    "class_name": CLASS_NAMES[image_index % len(CLASS_NAMES)],
                    "source": "rosendahl" if image_index % 2 else "vidir_modern",
                    "relative_path": relative_path,
                    "split": split,
                }
            )
            image_index += 1

        write_manifest(manifest_directory / f"{split}.csv", rows)

    return SyntheticDataset(root=root, manifest_directory=manifest_directory)
