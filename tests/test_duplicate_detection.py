"""Tests for HAM10000 duplicate detection."""

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from medvision.data.duplicates import (
    NEAR_PAIR_COLUMNS,
    build_hash_inventory,
    calculate_sha256,
    create_duplicate_summary,
    find_exact_duplicate_groups,
    find_near_duplicate_pairs,
    hamming_distance,
    save_duplicate_results,
)


def create_image(
    path: Path,
    color: tuple[int, int, int],
    quality: int = 95,
) -> None:
    """Create a small JPEG test image."""

    path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new(
        mode="RGB",
        size=(64, 64),
        color=color,
    )

    image.save(path, quality=quality)


def create_metadata(
    dataset_root: Path,
    rows: list[dict[str, object]],
) -> None:
    """Create a minimal HAM10000 metadata file."""

    dataset_root.mkdir(parents=True, exist_ok=True)

    metadata = pd.DataFrame(rows)
    metadata.to_csv(
        dataset_root / "HAM10000_metadata.csv",
        index=False,
    )


def metadata_row(
    image_id: str,
    lesion_id: str,
    class_name: str,
    source: str = "vidir_modern",
) -> dict[str, object]:
    """Create one valid HAM10000 metadata record."""

    return {
        "image_id": image_id,
        "lesion_id": lesion_id,
        "dx": class_name,
        "dataset": source,
    }


def test_sha256_matches_for_identical_files(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"

    create_image(first, color=(120, 80, 60))
    second.write_bytes(first.read_bytes())

    assert calculate_sha256(first) == calculate_sha256(second)


def test_hamming_distance() -> None:
    assert (
        hamming_distance(
            "0000000000000000",
            "0000000000000000",
        )
        == 0
    )

    assert (
        hamming_distance(
            "0000000000000000",
            "0000000000000001",
        )
        == 1
    )


def test_hash_inventory_uses_metadata(
    tmp_path: Path,
) -> None:
    create_metadata(
        tmp_path,
        [
            metadata_row(
                image_id="ISIC_0000001",
                lesion_id="HAM_0000001",
                class_name="mel",
                source="rosendahl",
            )
        ],
    )

    create_image(
        tmp_path / "images" / "ISIC_0000001.jpg",
        color=(120, 80, 60),
    )

    inventory = build_hash_inventory(tmp_path)

    assert len(inventory) == 1
    assert inventory.iloc[0]["image_id"] == "ISIC_0000001"
    assert inventory.iloc[0]["lesion_id"] == "HAM_0000001"
    assert inventory.iloc[0]["class_name"] == "mel"
    assert inventory.iloc[0]["source"] == "rosendahl"
    assert inventory.iloc[0]["sha256"]
    assert inventory.iloc[0]["perceptual_hash"]


def test_exact_cross_lesion_duplicate_is_detected(
    tmp_path: Path,
) -> None:
    create_metadata(
        tmp_path,
        [
            metadata_row(
                image_id="ISIC_0000001",
                lesion_id="HAM_0000001",
                class_name="mel",
            ),
            metadata_row(
                image_id="ISIC_0000002",
                lesion_id="HAM_0000002",
                class_name="mel",
            ),
        ],
    )

    first = tmp_path / "images" / "ISIC_0000001.jpg"
    second = tmp_path / "images" / "ISIC_0000002.jpg"

    create_image(first, color=(120, 80, 60))
    second.write_bytes(first.read_bytes())

    inventory = build_hash_inventory(tmp_path)
    groups = find_exact_duplicate_groups(inventory)

    assert len(groups) == 1
    assert groups[0]["image_count"] == 2
    assert groups[0]["cross_lesion"] is True
    assert groups[0]["cross_class"] is False
    assert groups[0]["cross_source"] is False


def test_exact_cross_class_duplicate_is_flagged(
    tmp_path: Path,
) -> None:
    create_metadata(
        tmp_path,
        [
            metadata_row(
                image_id="ISIC_0000001",
                lesion_id="HAM_0000001",
                class_name="mel",
            ),
            metadata_row(
                image_id="ISIC_0000002",
                lesion_id="HAM_0000002",
                class_name="nv",
            ),
        ],
    )

    first = tmp_path / "images" / "ISIC_0000001.jpg"
    second = tmp_path / "images" / "ISIC_0000002.jpg"

    create_image(first, color=(120, 80, 60))
    second.write_bytes(first.read_bytes())

    inventory = build_hash_inventory(tmp_path)
    groups = find_exact_duplicate_groups(inventory)

    assert len(groups) == 1
    assert groups[0]["cross_lesion"] is True
    assert groups[0]["cross_class"] is True


def test_exact_cross_source_duplicate_is_flagged(
    tmp_path: Path,
) -> None:
    create_metadata(
        tmp_path,
        [
            metadata_row(
                image_id="ISIC_0000001",
                lesion_id="HAM_0000001",
                class_name="mel",
                source="rosendahl",
            ),
            metadata_row(
                image_id="ISIC_0000002",
                lesion_id="HAM_0000001",
                class_name="mel",
                source="vidir_modern",
            ),
        ],
    )

    first = tmp_path / "images" / "ISIC_0000001.jpg"
    second = tmp_path / "images" / "ISIC_0000002.jpg"

    create_image(first, color=(120, 80, 60))
    second.write_bytes(first.read_bytes())

    inventory = build_hash_inventory(tmp_path)
    groups = find_exact_duplicate_groups(inventory)

    assert len(groups) == 1
    assert groups[0]["same_lesion"] is True
    assert groups[0]["cross_lesion"] is False
    assert groups[0]["cross_class"] is False
    assert groups[0]["cross_source"] is True


def test_near_duplicate_columns_and_flags_exist(
    tmp_path: Path,
) -> None:
    create_metadata(
        tmp_path,
        [
            metadata_row(
                image_id="ISIC_0000001",
                lesion_id="HAM_0000001",
                class_name="mel",
            ),
            metadata_row(
                image_id="ISIC_0000002",
                lesion_id="HAM_0000001",
                class_name="mel",
            ),
        ],
    )

    create_image(
        tmp_path / "images" / "ISIC_0000001.jpg",
        color=(120, 80, 60),
        quality=95,
    )

    create_image(
        tmp_path / "images" / "ISIC_0000002.jpg",
        color=(120, 80, 60),
        quality=75,
    )

    inventory = build_hash_inventory(tmp_path)

    # Confirm this test exercises near rather than exact duplication.
    assert inventory.loc[0, "sha256"] != inventory.loc[1, "sha256"]

    pairs = find_near_duplicate_pairs(
        inventory,
        maximum_distance=6,
    )

    assert len(pairs) == 1
    assert "hash_distance" in pairs.columns
    assert "same_lesion" in pairs.columns
    assert "cross_lesion" in pairs.columns
    assert "cross_class" in pairs.columns
    assert "cross_source" in pairs.columns

    assert bool(pairs.iloc[0]["same_lesion"]) is True
    assert bool(pairs.iloc[0]["cross_lesion"]) is False
    assert bool(pairs.iloc[0]["cross_class"]) is False


def test_hamming_distance_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        hamming_distance("0000000000000000", "00")


def test_hash_inventory_rejects_image_without_metadata(
    tmp_path: Path,
) -> None:
    create_metadata(
        tmp_path,
        [
            metadata_row(
                image_id="ISIC_0000001",
                lesion_id="HAM_0000001",
                class_name="mel",
            )
        ],
    )

    create_image(tmp_path / "images" / "ISIC_0000001.jpg", color=(120, 80, 60))
    create_image(tmp_path / "images" / "ISIC_0000002.jpg", color=(120, 80, 60))

    with pytest.raises(ValueError, match="no metadata record"):
        build_hash_inventory(tmp_path)


def test_near_duplicate_cross_lesion_class_and_source_are_flagged(
    tmp_path: Path,
) -> None:
    create_metadata(
        tmp_path,
        [
            metadata_row(
                image_id="ISIC_0000001",
                lesion_id="HAM_0000001",
                class_name="mel",
                source="rosendahl",
            ),
            metadata_row(
                image_id="ISIC_0000002",
                lesion_id="HAM_0000002",
                class_name="nv",
                source="vidir_modern",
            ),
        ],
    )

    create_image(tmp_path / "images" / "ISIC_0000001.jpg", color=(120, 80, 60), quality=95)
    create_image(tmp_path / "images" / "ISIC_0000002.jpg", color=(120, 80, 60), quality=75)

    inventory = build_hash_inventory(tmp_path)
    pairs = find_near_duplicate_pairs(inventory, maximum_distance=6)

    assert len(pairs) == 1

    pair = pairs.iloc[0]

    assert bool(pair["same_lesion"]) is False
    assert bool(pair["cross_lesion"]) is True
    assert bool(pair["cross_class"]) is True
    assert bool(pair["cross_source"]) is True


def test_near_duplicate_search_excludes_exact_duplicates(
    tmp_path: Path,
) -> None:
    create_metadata(
        tmp_path,
        [
            metadata_row(
                image_id="ISIC_0000001",
                lesion_id="HAM_0000001",
                class_name="mel",
            ),
            metadata_row(
                image_id="ISIC_0000002",
                lesion_id="HAM_0000001",
                class_name="mel",
            ),
        ],
    )

    first = tmp_path / "images" / "ISIC_0000001.jpg"
    create_image(first, color=(120, 80, 60))
    (tmp_path / "images" / "ISIC_0000002.jpg").write_bytes(first.read_bytes())

    inventory = build_hash_inventory(tmp_path)
    pairs = find_near_duplicate_pairs(inventory, maximum_distance=6)

    assert pairs.empty
    assert list(pairs.columns) == NEAR_PAIR_COLUMNS


def test_find_near_duplicate_pairs_rejects_invalid_distance() -> None:
    inventory = pd.DataFrame(columns=["image_id", "perceptual_hash"])

    with pytest.raises(ValueError, match="maximum_distance"):
        find_near_duplicate_pairs(inventory, maximum_distance=65)


def test_duplicate_summary_counts_same_and_cross_lesion(
    tmp_path: Path,
) -> None:
    create_metadata(
        tmp_path,
        [
            metadata_row(
                image_id="ISIC_0000001",
                lesion_id="HAM_0000001",
                class_name="mel",
            ),
            metadata_row(
                image_id="ISIC_0000002",
                lesion_id="HAM_0000001",
                class_name="mel",
            ),
            metadata_row(
                image_id="ISIC_0000003",
                lesion_id="HAM_0000002",
                class_name="nv",
            ),
        ],
    )

    first = tmp_path / "images" / "ISIC_0000001.jpg"
    create_image(first, color=(120, 80, 60))
    (tmp_path / "images" / "ISIC_0000002.jpg").write_bytes(first.read_bytes())
    create_image(tmp_path / "images" / "ISIC_0000003.jpg", color=(120, 80, 60), quality=75)

    inventory = build_hash_inventory(tmp_path)
    groups = find_exact_duplicate_groups(inventory)
    pairs = find_near_duplicate_pairs(inventory, maximum_distance=6)
    summary = create_duplicate_summary(groups, pairs)

    assert summary["exact_duplicate_groups"] == 1
    assert summary["images_in_exact_duplicate_groups"] == 2
    assert summary["exact_same_lesion_groups"] == 1
    assert summary["exact_cross_lesion_groups"] == 0
    assert summary["near_duplicate_pairs"] == 2
    assert summary["near_same_lesion_pairs"] == 0
    assert summary["near_cross_lesion_pairs"] == 2
    assert summary["near_cross_class_pairs"] == 2


def test_save_duplicate_results_writes_reports(
    tmp_path: Path,
) -> None:
    create_metadata(
        tmp_path,
        [
            metadata_row(
                image_id="ISIC_0000001",
                lesion_id="HAM_0000001",
                class_name="mel",
            )
        ],
    )
    create_image(tmp_path / "images" / "ISIC_0000001.jpg", color=(120, 80, 60))

    inventory = build_hash_inventory(tmp_path)
    groups = find_exact_duplicate_groups(inventory)
    pairs = find_near_duplicate_pairs(inventory)
    output_directory = tmp_path / "reports"

    save_duplicate_results(inventory, groups, pairs, output_directory)

    summary = json.loads((output_directory / "duplicate_summary.json").read_text())

    assert summary["exact_duplicate_groups"] == 0
    assert summary["near_duplicate_pairs"] == 0
    assert (output_directory / "hash_inventory.csv").exists()
    assert (output_directory / "near_duplicate_pairs.csv").exists()
    assert (output_directory / "exact_duplicate_groups.json").exists()
