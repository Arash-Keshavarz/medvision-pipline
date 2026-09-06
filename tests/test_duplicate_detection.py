from pathlib import Path

from PIL import Image

from medvision.data.duplicates import (
    build_hash_inventory,
    calculate_sha256,
    find_exact_duplicate_groups,
    find_near_duplicate_pairs,
    hamming_distance,
)


def create_image(
    path: Path,
    color: tuple[int, int, int],
    quality: int = 95,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new(
        mode="RGB",
        size=(64, 64),
        color=color,
    )

    image.save(path, quality=quality)


def test_sha256_matches_for_identical_files(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"

    create_image(first, color=(120, 80, 60))
    second.write_bytes(first.read_bytes())

    assert calculate_sha256(first) == calculate_sha256(second)


def test_hamming_distance() -> None:
    assert hamming_distance("0000000000000000", "0000000000000000") == 0
    assert hamming_distance("0000000000000000", "0000000000000001") == 1


def test_exact_duplicate_group_is_detected(
    tmp_path: Path,
) -> None:
    first = tmp_path / "Train" / "melanoma" / "first.jpg"
    second = tmp_path / "Test" / "melanoma" / "second.jpg"

    create_image(first, color=(120, 80, 60))
    second.parent.mkdir(parents=True)
    second.write_bytes(first.read_bytes())

    inventory = build_hash_inventory(tmp_path)
    groups = find_exact_duplicate_groups(inventory)

    assert len(groups) == 1
    assert groups[0]["image_count"] == 2
    assert groups[0]["cross_split"] is True
    assert groups[0]["cross_class"] is False


def test_cross_class_duplicate_is_flagged(
    tmp_path: Path,
) -> None:
    first = tmp_path / "Train" / "melanoma" / "first.jpg"
    second = tmp_path / "Train" / "nevus" / "second.jpg"

    create_image(first, color=(120, 80, 60))
    second.parent.mkdir(parents=True)
    second.write_bytes(first.read_bytes())

    inventory = build_hash_inventory(tmp_path)
    groups = find_exact_duplicate_groups(inventory)

    assert len(groups) == 1
    assert groups[0]["cross_class"] is True


def test_near_duplicate_columns_exist(
    tmp_path: Path,
) -> None:
    create_image(
        tmp_path / "Train" / "melanoma" / "first.jpg",
        color=(120, 80, 60),
        quality=95,
    )
    create_image(
        tmp_path / "Test" / "melanoma" / "second.jpg",
        color=(120, 80, 60),
        quality=75,
    )

    inventory = build_hash_inventory(tmp_path)
    pairs = find_near_duplicate_pairs(
        inventory,
        maximum_distance=6,
    )

    assert "hash_distance" in pairs.columns
    assert "cross_split" in pairs.columns
    assert "cross_class" in pairs.columns
