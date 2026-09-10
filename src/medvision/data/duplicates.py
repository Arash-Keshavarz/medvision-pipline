"""Exact and perceptual duplicate detection for HAM10000."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imagehash
import pandas as pd
from PIL import Image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

NEAR_PAIR_COLUMNS = [
    "image_id_left",
    "image_id_right",
    "lesion_id_left",
    "lesion_id_right",
    "class_left",
    "class_right",
    "source_left",
    "source_right",
    "path_left",
    "path_right",
    "hash_distance",
    "same_lesion",
    "cross_lesion",
    "cross_class",
    "cross_source",
]


def calculate_sha256(image_path: Path) -> str:
    """Calculate the SHA-256 digest of a file."""

    digest = hashlib.sha256()

    with image_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def calculate_perceptual_hash(image_path: Path) -> str:
    """Calculate a 64-bit perceptual hash."""

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        return str(imagehash.phash(image))


def hamming_distance(first_hash: str, second_hash: str) -> int:
    """Calculate the Hamming distance between hexadecimal hashes."""

    return (int(first_hash, 16) ^ int(second_hash, 16)).bit_count()


def build_hash_inventory(dataset_root: Path) -> pd.DataFrame:
    """Build an image-hash inventory using HAM10000 metadata."""

    dataset_root = dataset_root.resolve()
    image_root = dataset_root / "images"
    metadata_path = dataset_root / "HAM10000_metadata.csv"

    if not image_root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    required_columns = {
        "image_id",
        "lesion_id",
        "dx",
        "dataset",
    }

    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing_columns)}")

    if metadata["image_id"].duplicated().any():
        raise ValueError("Metadata contains duplicate image_id values.")

    metadata_by_id = metadata.set_index("image_id")

    image_paths = sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(f"No supported images found under: {image_root}")

    records: list[dict[str, Any]] = []

    for index, image_path in enumerate(image_paths, start=1):
        image_id = image_path.stem

        if image_id not in metadata_by_id.index:
            raise ValueError(f"Image has no metadata record: {image_id}")

        metadata_row = metadata_by_id.loc[image_id]

        records.append(
            {
                "image_id": image_id,
                "lesion_id": metadata_row["lesion_id"],
                "class_name": metadata_row["dx"],
                "source": metadata_row["dataset"],
                "relative_path": str(image_path.relative_to(dataset_root)),
                "sha256": calculate_sha256(image_path),
                "perceptual_hash": calculate_perceptual_hash(image_path),
            }
        )

        if index % 1000 == 0:
            print(f"Hashed {index:,}/{len(image_paths):,} images")

    inventory = pd.DataFrame(records)

    metadata_ids = set(metadata["image_id"])
    discovered_ids = set(inventory["image_id"])
    missing_images = metadata_ids - discovered_ids

    if missing_images:
        raise ValueError(f"{len(missing_images)} metadata records have no image.")

    return inventory.sort_values("image_id").reset_index(drop=True)


def find_exact_duplicate_groups(
    inventory: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Find files with identical SHA-256 hashes."""

    duplicate_groups: list[dict[str, Any]] = []

    for sha256, group in inventory.groupby(
        "sha256",
        sort=True,
    ):
        if len(group) < 2:
            continue

        lesion_ids = sorted(group["lesion_id"].astype(str).unique().tolist())
        classes = sorted(group["class_name"].astype(str).unique().tolist())
        sources = sorted(group["source"].astype(str).unique().tolist())

        images = []

        for row in group.sort_values("image_id").itertuples():
            images.append(
                {
                    "image_id": row.image_id,
                    "lesion_id": row.lesion_id,
                    "class_name": row.class_name,
                    "source": row.source,
                    "relative_path": row.relative_path,
                }
            )

        duplicate_groups.append(
            {
                "sha256": sha256,
                "image_count": int(len(group)),
                "lesion_ids": lesion_ids,
                "classes": classes,
                "sources": sources,
                "cross_lesion": len(lesion_ids) > 1,
                "cross_class": len(classes) > 1,
                "cross_source": len(sources) > 1,
                "images": images,
            }
        )

    return duplicate_groups


@dataclass
class _BKNode:
    """Node used for bounded Hamming-distance search."""

    value: int
    row_indices: list[int] = field(default_factory=list)
    children: dict[int, "_BKNode"] = field(default_factory=dict)

    def add(self, value: int, row_index: int) -> None:
        """Add a hash value to the tree."""

        distance = (self.value ^ value).bit_count()

        if distance == 0:
            self.row_indices.append(row_index)
            return

        child = self.children.get(distance)

        if child is None:
            self.children[distance] = _BKNode(
                value=value,
                row_indices=[row_index],
            )
            return

        child.add(value, row_index)

    def query(
        self,
        value: int,
        maximum_distance: int,
    ) -> list[tuple[int, int]]:
        """Return matching row indices and their distances."""

        distance = (self.value ^ value).bit_count()
        matches: list[tuple[int, int]] = []

        if distance <= maximum_distance:
            matches.extend((row_index, distance) for row_index in self.row_indices)

        minimum_edge = distance - maximum_distance
        maximum_edge = distance + maximum_distance

        for edge, child in self.children.items():
            if minimum_edge <= edge <= maximum_edge:
                matches.extend(child.query(value, maximum_distance))

        return matches


def find_near_duplicate_pairs(
    inventory: pd.DataFrame,
    maximum_distance: int = 6,
) -> pd.DataFrame:
    """Find perceptually similar image pairs using a BK-tree."""

    if not 0 <= maximum_distance <= 64:
        raise ValueError("maximum_distance must be between 0 and 64.")

    ordered = inventory.sort_values("image_id").reset_index(drop=True)

    if ordered.empty:
        return pd.DataFrame(columns=NEAR_PAIR_COLUMNS)

    root: _BKNode | None = None
    pairs: list[dict[str, Any]] = []

    for right_index, right in ordered.iterrows():
        hash_value = int(right["perceptual_hash"], 16)

        if root is not None:
            candidates = root.query(
                hash_value,
                maximum_distance,
            )

            for left_index, distance in candidates:
                left = ordered.iloc[left_index]

                # Exact byte duplicates are reported separately.
                if left["sha256"] == right["sha256"]:
                    continue

                same_lesion = left["lesion_id"] == right["lesion_id"]

                pairs.append(
                    {
                        "image_id_left": left["image_id"],
                        "image_id_right": right["image_id"],
                        "lesion_id_left": left["lesion_id"],
                        "lesion_id_right": right["lesion_id"],
                        "class_left": left["class_name"],
                        "class_right": right["class_name"],
                        "source_left": left["source"],
                        "source_right": right["source"],
                        "path_left": left["relative_path"],
                        "path_right": right["relative_path"],
                        "hash_distance": int(distance),
                        "same_lesion": bool(same_lesion),
                        "cross_lesion": bool(not same_lesion),
                        "cross_class": bool(left["class_name"] != right["class_name"]),
                        "cross_source": bool(left["source"] != right["source"]),
                    }
                )

        if root is None:
            root = _BKNode(
                value=hash_value,
                row_indices=[right_index],
            )
        else:
            root.add(hash_value, right_index)

    if not pairs:
        return pd.DataFrame(columns=NEAR_PAIR_COLUMNS)

    return (
        pd.DataFrame(pairs, columns=NEAR_PAIR_COLUMNS)
        .sort_values(
            [
                "hash_distance",
                "image_id_left",
                "image_id_right",
            ]
        )
        .reset_index(drop=True)
    )


def create_duplicate_summary(
    exact_groups: list[dict[str, Any]],
    near_pairs: pd.DataFrame,
) -> dict[str, int]:
    """Create a summary of exact and near duplicates."""

    images_in_exact_groups = sum(group["image_count"] for group in exact_groups)

    return {
        "exact_duplicate_groups": len(exact_groups),
        "images_in_exact_duplicate_groups": (images_in_exact_groups),
        "exact_cross_lesion_groups": sum(bool(group["cross_lesion"]) for group in exact_groups),
        "exact_cross_class_groups": sum(bool(group["cross_class"]) for group in exact_groups),
        "exact_cross_source_groups": sum(bool(group["cross_source"]) for group in exact_groups),
        "near_duplicate_pairs": int(len(near_pairs)),
        "near_same_lesion_pairs": int(near_pairs["same_lesion"].sum()),
        "near_cross_lesion_pairs": int(near_pairs["cross_lesion"].sum()),
        "near_cross_class_pairs": int(near_pairs["cross_class"].sum()),
        "near_cross_source_pairs": int(near_pairs["cross_source"].sum()),
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
