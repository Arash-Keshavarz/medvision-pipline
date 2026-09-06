"""Exact and perceptual duplicate detection."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import imagehash
import pandas as pd
from PIL import Image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def calculate_sha256(path: Path) -> str:
    """Calculate a file's SHA-256 hash."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_perceptual_hash(path: Path) -> str:
    """Calculate a perceptual hash for an image."""

    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        return str(imagehash.phash(rgb_image))


def discover_images(dataset_root: Path) -> list[dict[str, Any]]:
    """Discover images organized as split/class/image."""

    dataset_root = dataset_root.resolve()
    records: list[dict[str, Any]] = []

    for split_directory in sorted(dataset_root.iterdir()):
        if not split_directory.is_dir():
            continue

        for class_directory in sorted(split_directory.iterdir()):
            if not class_directory.is_dir():
                continue

            for image_path in sorted(class_directory.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    records.append(
                        {
                            "path": image_path,
                            "relative_path": image_path.relative_to(dataset_root).as_posix(),
                            "split": split_directory.name,
                            "class_name": class_directory.name,
                        }
                    )

    return records


def build_hash_inventory(dataset_root: Path) -> pd.DataFrame:
    """Calculate exact and perceptual hashes for every image."""

    records = discover_images(dataset_root)

    if not records:
        raise ValueError(f"No supported images found under {dataset_root}")

    inventory: list[dict[str, Any]] = []

    for index, record in enumerate(records, start=1):
        image_path = record["path"]

        inventory.append(
            {
                "relative_path": record["relative_path"],
                "split": record["split"],
                "class_name": record["class_name"],
                "sha256": calculate_sha256(image_path),
                "perceptual_hash": calculate_perceptual_hash(image_path),
            }
        )

        if index % 250 == 0:
            print(f"Hashed {index}/{len(records)} images")

    return pd.DataFrame(inventory)


def find_exact_duplicate_groups(
    inventory: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Find groups with identical SHA-256 hashes."""

    groups: list[dict[str, Any]] = []

    for sha256, group in inventory.groupby("sha256"):
        if len(group) < 2:
            continue

        records = group.to_dict(orient="records")

        splits = sorted(group["split"].unique())
        classes = sorted(group["class_name"].unique())

        groups.append(
            {
                "sha256": sha256,
                "image_count": len(records),
                "cross_split": len(splits) > 1,
                "cross_class": len(classes) > 1,
                "splits": splits,
                "classes": classes,
                "images": [
                    {
                        "relative_path": record["relative_path"],
                        "split": record["split"],
                        "class_name": record["class_name"],
                    }
                    for record in records
                ],
            }
        )

    return groups


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Calculate Hamming distance between hexadecimal hashes."""

    integer_a = int(hash_a, 16)
    integer_b = int(hash_b, 16)

    return (integer_a ^ integer_b).bit_count()


def find_near_duplicate_pairs(
    inventory: pd.DataFrame,
    maximum_distance: int = 6,
) -> pd.DataFrame:
    """Find visually similar images with different file hashes."""

    records = inventory.to_dict(orient="records")
    candidates: list[dict[str, Any]] = []

    for first, second in combinations(records, 2):
        # Exact binary duplicates are already reported separately.
        if first["sha256"] == second["sha256"]:
            continue

        distance = hamming_distance(
            first["perceptual_hash"],
            second["perceptual_hash"],
        )

        if distance <= maximum_distance:
            candidates.append(
                {
                    "first_path": first["relative_path"],
                    "second_path": second["relative_path"],
                    "first_split": first["split"],
                    "second_split": second["split"],
                    "first_class": first["class_name"],
                    "second_class": second["class_name"],
                    "hash_distance": distance,
                    "cross_split": (first["split"] != second["split"]),
                    "cross_class": (first["class_name"] != second["class_name"]),
                }
            )

    columns = [
        "first_path",
        "second_path",
        "first_split",
        "second_split",
        "first_class",
        "second_class",
        "hash_distance",
        "cross_split",
        "cross_class",
    ]

    return pd.DataFrame(candidates, columns=columns)


def create_duplicate_summary(
    exact_groups: list[dict[str, Any]],
    near_pairs: pd.DataFrame,
) -> dict[str, int]:
    """Summarize duplicate-detection results."""

    exact_image_count = sum(group["image_count"] for group in exact_groups)

    return {
        "exact_duplicate_groups": len(exact_groups),
        "images_in_exact_duplicate_groups": exact_image_count,
        "exact_cross_split_groups": sum(bool(group["cross_split"]) for group in exact_groups),
        "exact_cross_class_groups": sum(bool(group["cross_class"]) for group in exact_groups),
        "near_duplicate_pairs": len(near_pairs),
        "near_cross_split_pairs": (
            int(near_pairs["cross_split"].sum()) if not near_pairs.empty else 0
        ),
        "near_cross_class_pairs": (
            int(near_pairs["cross_class"].sum()) if not near_pairs.empty else 0
        ),
    }


def save_duplicate_results(
    inventory: pd.DataFrame,
    exact_groups: list[dict[str, Any]],
    near_pairs: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Save duplicate reports."""

    output_directory.mkdir(parents=True, exist_ok=True)

    inventory.to_csv(
        output_directory / "hash_inventory.csv",
        index=False,
    )

    near_pairs.to_csv(
        output_directory / "near_duplicate_pairs.csv",
        index=False,
    )

    with (output_directory / "exact_duplicate_groups.json").open("w", encoding="utf-8") as file:
        json.dump(exact_groups, file, indent=2)

    summary = create_duplicate_summary(
        exact_groups=exact_groups,
        near_pairs=near_pairs,
    )

    with (output_directory / "duplicate_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
