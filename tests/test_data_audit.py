"""Tests for HAM10000 dataset auditing."""

from pathlib import Path

import pandas as pd
from PIL import Image

from medvision.data.audit import (
    audit_dataset,
    create_class_summary,
    create_lesion_summary,
    create_source_summary,
    create_summary,
    save_audit_results,
    validate_structure,
)


def create_test_image(
    path: Path,
    size: tuple[int, int] = (600, 450),
) -> None:
    """Create a small valid RGB test image."""

    path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new(
        mode="RGB",
        size=size,
        color=(120, 80, 60),
    )
    image.save(path)


def create_test_metadata(
    dataset_root: Path,
    rows: list[dict[str, object]],
) -> None:
    """Create a HAM10000-style metadata file."""

    dataset_root.mkdir(parents=True, exist_ok=True)

    metadata = pd.DataFrame(rows)
    metadata.to_csv(
        dataset_root / "HAM10000_metadata.csv",
        index=False,
    )


def sample_metadata_rows() -> list[dict[str, object]]:
    """Return valid metadata rows for a small test dataset."""

    return [
        {
            "lesion_id": "HAM_0000001",
            "image_id": "ISIC_0000001",
            "dx": "mel",
            "dx_type": "histo",
            "age": 55,
            "sex": "male",
            "localization": "back",
            "dataset": "vidir_modern",
        },
        {
            "lesion_id": "HAM_0000002",
            "image_id": "ISIC_0000002",
            "dx": "nv",
            "dx_type": "follow_up",
            "age": 40,
            "sex": "female",
            "localization": "leg",
            "dataset": "rosendahl",
        },
    ]


def create_test_dataset(dataset_root: Path) -> None:
    """Create a minimal valid HAM10000 dataset."""

    create_test_metadata(
        dataset_root,
        sample_metadata_rows(),
    )

    create_test_image(
        dataset_root / "images" / "ISIC_0000001.jpg"
    )
    create_test_image(
        dataset_root / "images" / "ISIC_0000002.jpg",
        size=(300, 200),
    )


def test_audit_dataset_discovers_images_and_metadata(
    tmp_path: Path,
) -> None:
    create_test_dataset(tmp_path)

    inventory = audit_dataset(tmp_path)

    assert len(inventory) == 2
    assert inventory["readable"].all()
    assert inventory["has_metadata"].all()

    assert set(inventory["image_id"]) == {
        "ISIC_0000001",
        "ISIC_0000002",
    }

    assert set(inventory["class_name"]) == {"mel", "nv"}
    assert set(inventory["source"]) == {
        "vidir_modern",
        "rosendahl",
    }


def test_audit_dataset_records_image_dimensions(
    tmp_path: Path,
) -> None:
    create_test_dataset(tmp_path)

    inventory = audit_dataset(tmp_path)

    dimensions = {
        row.image_id: (row.width, row.height)
        for row in inventory.itertuples()
    }

    assert dimensions["ISIC_0000001"] == (600, 450)
    assert dimensions["ISIC_0000002"] == (300, 200)


def test_audit_dataset_detects_unreadable_image(
    tmp_path: Path,
) -> None:
    rows = sample_metadata_rows()[:1]
    create_test_metadata(tmp_path, rows)

    bad_image = tmp_path / "images" / "ISIC_0000001.jpg"
    bad_image.parent.mkdir(parents=True, exist_ok=True)
    bad_image.write_text("not an actual image", encoding="utf-8")

    inventory = audit_dataset(tmp_path)

    assert len(inventory) == 1
    assert not bool(inventory.iloc[0]["readable"])
    assert inventory.iloc[0]["error"] is not None


def test_audit_dataset_detects_image_without_metadata(
    tmp_path: Path,
) -> None:
    create_test_metadata(
        tmp_path,
        sample_metadata_rows()[:1],
    )

    create_test_image(
        tmp_path / "images" / "ISIC_0000001.jpg"
    )
    create_test_image(
        tmp_path / "images" / "ISIC_9999999.jpg"
    )

    inventory = audit_dataset(tmp_path)
    summary = create_summary(inventory)

    assert summary["total_images"] == 2
    assert summary["images_without_metadata"] == 1


def test_audit_dataset_detects_metadata_without_image(
    tmp_path: Path,
) -> None:
    create_test_metadata(
        tmp_path,
        sample_metadata_rows(),
    )

    create_test_image(
        tmp_path / "images" / "ISIC_0000001.jpg"
    )

    inventory = audit_dataset(tmp_path)
    summary = create_summary(inventory)

    assert summary["metadata_without_images"] == 1


def test_create_summary(
    tmp_path: Path,
) -> None:
    create_test_dataset(tmp_path)

    inventory = audit_dataset(tmp_path)
    summary = create_summary(inventory)

    assert summary["total_images"] == 2
    assert summary["readable_images"] == 2
    assert summary["unreadable_images"] == 0
    assert summary["unique_image_ids"] == 2
    assert summary["unique_lesions"] == 2
    assert summary["class_counts"] == {
        "mel": 1,
        "nv": 1,
    }


def test_create_class_summary(
    tmp_path: Path,
) -> None:
    rows = sample_metadata_rows()
    rows.append(
        {
            **rows[0],
            "image_id": "ISIC_0000003",
            "lesion_id": "HAM_0000003",
        }
    )

    create_test_metadata(tmp_path, rows)

    for image_id in (
        "ISIC_0000001",
        "ISIC_0000002",
        "ISIC_0000003",
    ):
        create_test_image(
            tmp_path / "images" / f"{image_id}.jpg"
        )

    inventory = audit_dataset(tmp_path)
    class_summary = create_class_summary(inventory)

    counts = dict(
        zip(
            class_summary["class_name"],
            class_summary["image_count"],
            strict=True,
        )
    )

    assert counts == {
        "mel": 2,
        "nv": 1,
    }


def test_create_source_summary(
    tmp_path: Path,
) -> None:
    create_test_dataset(tmp_path)

    inventory = audit_dataset(tmp_path)
    source_summary = create_source_summary(inventory)

    assert set(source_summary["source"]) == {
        "vidir_modern",
        "rosendahl",
    }
    assert source_summary["image_count"].sum() == 2


def test_create_lesion_summary_groups_related_images(
    tmp_path: Path,
) -> None:
    rows = sample_metadata_rows()
    rows.append(
        {
            **rows[0],
            "image_id": "ISIC_0000003",
        }
    )

    create_test_metadata(tmp_path, rows)

    for image_id in (
        "ISIC_0000001",
        "ISIC_0000002",
        "ISIC_0000003",
    ):
        create_test_image(
            tmp_path / "images" / f"{image_id}.jpg"
        )

    inventory = audit_dataset(tmp_path)
    lesion_summary = create_lesion_summary(inventory)

    lesion = lesion_summary.loc[
        lesion_summary["lesion_id"] == "HAM_0000001"
    ].iloc[0]

    assert lesion["image_count"] == 2
    assert lesion["class_count"] == 1


def test_validate_structure_detects_unexpected_class(
    tmp_path: Path,
) -> None:
    rows = sample_metadata_rows()[:1]
    rows[0]["dx"] = "unknown_class"

    create_test_metadata(tmp_path, rows)
    create_test_image(
        tmp_path / "images" / "ISIC_0000001.jpg"
    )

    inventory = audit_dataset(tmp_path)
    validation = validate_structure(inventory)

    assert validation["unexpected_classes"] == ["unknown_class"]
    assert "mel" in validation["missing_classes"]


def test_save_audit_results(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    output_directory = tmp_path / "reports"

    create_test_dataset(dataset_root)

    inventory = audit_dataset(dataset_root)
    save_audit_results(
        inventory=inventory,
        output_directory=output_directory,
    )

    expected_files = {
        "audit_summary.json",
        "image_inventory.csv",
        "class_summary.csv",
        "source_summary.csv",
        "lesion_summary.csv",
        "class_distribution.png",
        "source_distribution.png",
        "image_dimensions.png",
    }

    discovered_files = {
        path.name
        for path in output_directory.iterdir()
        if path.is_file()
    }

    assert expected_files <= discovered_files