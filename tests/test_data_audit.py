from pathlib import Path

from PIL import Image

from medvision.data.audit import (
    audit_dataset,
    create_summary,
    validate_structure,
    create_class_balance_summary,
    create_dimension_summary,
)


def create_test_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new(
        mode="RGB",
        size=(32, 24),
        color=(120, 80, 60),
    )
    image.save(path)


def test_audit_discovers_images(tmp_path: Path) -> None:
    create_test_image(tmp_path / "Train" / "melanoma" / "image_1.jpg")
    create_test_image(tmp_path / "Test" / "nevus" / "image_2.png")

    inventory = audit_dataset(tmp_path)

    assert len(inventory) == 2
    assert set(inventory["split"]) == {"Train", "Test"}
    assert set(inventory["class_name"]) == {"melanoma", "nevus"}
    assert inventory["readable"].all()


def test_audit_records_image_dimensions(tmp_path: Path) -> None:
    create_test_image(tmp_path / "Train" / "melanoma" / "image.jpg")

    inventory = audit_dataset(tmp_path)
    record = inventory.iloc[0]

    assert record["width"] == 32
    assert record["height"] == 24
    assert record["mode"] == "RGB"


def test_audit_detects_corrupted_image(tmp_path: Path) -> None:
    corrupted_path = tmp_path / "Train" / "melanoma" / "corrupted.jpg"
    corrupted_path.parent.mkdir(parents=True)
    corrupted_path.write_text("This is not an image.")

    inventory = audit_dataset(tmp_path)

    assert len(inventory) == 1
    assert not bool(inventory.iloc[0]["readable"])
    assert inventory.iloc[0]["error"] is not None


def test_summary_contains_image_count(tmp_path: Path) -> None:
    create_test_image(tmp_path / "Train" / "melanoma" / "image.jpg")

    inventory = audit_dataset(tmp_path)
    summary = create_summary(inventory)

    assert summary["total_images"] == 1
    assert summary["readable_images"] == 1
    assert summary["unreadable_images"] == 0


def test_structure_validation_reports_missing_classes(
    tmp_path: Path,
) -> None:
    create_test_image(tmp_path / "Train" / "melanoma" / "image.jpg")

    inventory = audit_dataset(tmp_path)
    validation = validate_structure(inventory)

    assert validation["discovered_splits"] == ["Train"]
    assert validation["missing_splits"] == ["Test"]
    assert "nevus" in validation["missing_classes"]


def test_dimension_summary_is_grouped_by_split(
    tmp_path: Path,
) -> None:
    create_test_image(tmp_path / "Train" / "melanoma" / "train.jpg")
    create_test_image(tmp_path / "Test" / "melanoma" / "test.jpg")

    inventory = audit_dataset(tmp_path)
    summary = create_dimension_summary(inventory)

    assert set(summary["split"]) == {"Train", "Test"}
    assert summary["image_count"].sum() == 2


def test_class_balance_summary_calculates_ratio(
    tmp_path: Path,
) -> None:
    create_test_image(tmp_path / "Train" / "melanoma" / "image_1.jpg")
    create_test_image(tmp_path / "Train" / "melanoma" / "image_2.jpg")
    create_test_image(tmp_path / "Train" / "nevus" / "image_3.jpg")

    inventory = audit_dataset(tmp_path)
    summary = create_class_balance_summary(inventory)

    assert summary["maximum_class"] == "melanoma"
    assert summary["minimum_class"] == "nevus"
    assert summary["imbalance_ratio"] == 2.0
