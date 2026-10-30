"""Tests for lesion-aware HAM10000 splitting."""

import json
from pathlib import Path

import pandas as pd
import pytest

from medvision.data.constants import EXPECTED_CLASSES
from medvision.data.splits import (
    SPLIT_COLUMNS,
    SPLIT_NAMES,
    SplitConfig,
    create_lesion_aware_splits,
    load_metadata,
    save_splits,
    validate_metadata,
    validate_splits,
)

SOURCES = ["rosendahl", "vidir_modern", "vidir_molemax", "vienna_dias"]
LESIONS_PER_CLASS = 24


def build_metadata_rows() -> list[dict[str, object]]:
    """Create synthetic HAM10000 metadata with multi-image lesions."""

    rows: list[dict[str, object]] = []
    image_number = 0

    for class_name in sorted(EXPECTED_CLASSES):
        for lesion_number in range(LESIONS_PER_CLASS):
            lesion_id = f"HAM_{class_name}_{lesion_number:03d}"
            images_in_lesion = 2 if lesion_number % 3 == 0 else 1

            for _ in range(images_in_lesion):
                rows.append(
                    {
                        "lesion_id": lesion_id,
                        "image_id": f"ISIC_{image_number:07d}",
                        "dx": class_name,
                        "dataset": SOURCES[lesion_number % len(SOURCES)],
                    }
                )
                image_number += 1

    return rows


def create_dataset(
    dataset_root: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write metadata and placeholder image files."""

    image_root = dataset_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(dataset_root / "HAM10000_metadata.csv", index=False)

    # Splitting only needs the files to exist; image content is irrelevant.
    for row in rows:
        (image_root / f"{row['image_id']}.jpg").write_bytes(b"")


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "ham10000"
    create_dataset(root, build_metadata_rows())
    return root


@pytest.fixture
def metadata(dataset_root: Path) -> pd.DataFrame:
    return load_metadata(dataset_root)


def test_load_metadata_adds_relative_paths(metadata: pd.DataFrame) -> None:
    assert list(metadata.columns) == SPLIT_COLUMNS[:-1]
    assert metadata.iloc[0]["relative_path"] == f"images/{metadata.iloc[0]['image_id']}.jpg"


def test_splits_assign_every_image_once(metadata: pd.DataFrame) -> None:
    split_data = create_lesion_aware_splits(metadata)

    assert len(split_data) == len(metadata)
    assert set(split_data["image_id"]) == set(metadata["image_id"])
    assert not split_data["image_id"].duplicated().any()
    assert split_data["lesion_id"].nunique() == metadata["lesion_id"].nunique()


def test_splits_have_no_lesion_leakage(metadata: pd.DataFrame) -> None:
    split_data = create_lesion_aware_splits(metadata)

    assert split_data.groupby("lesion_id")["split"].nunique().max() == 1


def test_all_splits_and_classes_are_present(metadata: pd.DataFrame) -> None:
    split_data = create_lesion_aware_splits(metadata)

    assert set(split_data["split"]) == set(SPLIT_NAMES)

    for _, group in split_data.groupby("split"):
        assert set(group["class_name"]) == EXPECTED_CLASSES


def test_split_ratios_are_approximately_70_15_15(metadata: pd.DataFrame) -> None:
    split_data = create_lesion_aware_splits(metadata)
    lesion_ratios = (
        split_data.groupby("split")["lesion_id"].nunique() / split_data["lesion_id"].nunique()
    )

    assert lesion_ratios["train"] == pytest.approx(0.70, abs=0.05)
    assert lesion_ratios["validation"] == pytest.approx(0.15, abs=0.05)
    assert lesion_ratios["test"] == pytest.approx(0.15, abs=0.05)


def test_splits_are_deterministic(metadata: pd.DataFrame) -> None:
    first = create_lesion_aware_splits(metadata, SplitConfig(random_seed=7))
    second = create_lesion_aware_splits(metadata, SplitConfig(random_seed=7))

    pd.testing.assert_frame_equal(first, second)


def test_splits_do_not_depend_on_input_order_or_index(metadata: pd.DataFrame) -> None:
    reordered = metadata.sample(frac=1.0, random_state=0)
    reordered.index = reordered.index + 1000

    expected = create_lesion_aware_splits(metadata)
    actual = create_lesion_aware_splits(reordered)

    pd.testing.assert_frame_equal(expected, actual)


def test_different_seeds_produce_different_valid_splits(metadata: pd.DataFrame) -> None:
    first = create_lesion_aware_splits(metadata, SplitConfig(random_seed=1))
    second = create_lesion_aware_splits(metadata, SplitConfig(random_seed=2))

    validate_splits(second)
    assert not first["split"].equals(second["split"])


def test_exact_duplicates_sharing_a_lesion_stay_together(tmp_path: Path) -> None:
    rows = build_metadata_rows()
    duplicate_lesion = rows[0]["lesion_id"]
    duplicate_ids = [row["image_id"] for row in rows if row["lesion_id"] == duplicate_lesion]
    create_dataset(tmp_path, rows)

    # Make the lesion's two images byte-identical, as in HAM10000.
    image_root = tmp_path / "images"
    (image_root / f"{duplicate_ids[1]}.jpg").write_bytes(b"same bytes")
    (image_root / f"{duplicate_ids[0]}.jpg").write_bytes(b"same bytes")

    for seed in range(5):
        split_data = create_lesion_aware_splits(
            load_metadata(tmp_path), SplitConfig(random_seed=seed)
        )
        duplicate_splits = split_data.loc[split_data["image_id"].isin(duplicate_ids), "split"]

        assert len(duplicate_ids) == 2
        assert duplicate_splits.nunique() == 1


def test_conflicting_lesion_labels_fail(metadata: pd.DataFrame) -> None:
    conflicting = metadata.copy()
    multi_image_lesion = conflicting["lesion_id"][conflicting["lesion_id"].duplicated()].iloc[0]
    row = conflicting.index[conflicting["lesion_id"] == multi_image_lesion][0]
    current_class = conflicting.loc[row, "class_name"]
    conflicting.loc[row, "class_name"] = "nv" if current_class != "nv" else "mel"

    with pytest.raises(ValueError, match="multiple diagnosis labels"):
        create_lesion_aware_splits(conflicting)


def test_duplicate_image_ids_fail(metadata: pd.DataFrame) -> None:
    duplicated = pd.concat([metadata, metadata.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate image IDs"):
        validate_metadata(duplicated)


def test_unexpected_class_fails(metadata: pd.DataFrame) -> None:
    invalid = metadata.copy()
    invalid.loc[invalid.index[0], "class_name"] = "unknown"

    with pytest.raises(ValueError, match="Unexpected classes"):
        validate_metadata(invalid)


def test_missing_values_fail(metadata: pd.DataFrame) -> None:
    invalid = metadata.copy()
    invalid.loc[invalid.index[0], "lesion_id"] = None

    with pytest.raises(ValueError, match="missing values"):
        validate_metadata(invalid)


@pytest.mark.parametrize("column", sorted(["image_id", "lesion_id", "dx", "dataset"]))
def test_missing_required_metadata_column_fails(tmp_path: Path, column: str) -> None:
    rows = [
        {key: value for key, value in row.items() if key != column} for row in build_metadata_rows()
    ]
    (tmp_path / "images").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "HAM10000_metadata.csv", index=False)

    with pytest.raises(ValueError, match="Missing metadata columns"):
        load_metadata(tmp_path)


def test_missing_image_file_fails(dataset_root: Path) -> None:
    next((dataset_root / "images").glob("*.jpg")).unlink()

    with pytest.raises(FileNotFoundError, match="metadata images are missing"):
        load_metadata(dataset_root)


def test_missing_metadata_file_fails(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()

    with pytest.raises(FileNotFoundError, match="Metadata file not found"):
        load_metadata(tmp_path)


def test_validate_splits_detects_lesion_leakage(metadata: pd.DataFrame) -> None:
    split_data = create_lesion_aware_splits(metadata)
    multi_image_lesion = split_data["lesion_id"][split_data["lesion_id"].duplicated()].iloc[0]
    row = split_data.index[split_data["lesion_id"] == multi_image_lesion][0]
    current_split = split_data.loc[row, "split"]
    split_data.loc[row, "split"] = "test" if current_split != "test" else "train"

    with pytest.raises(ValueError, match="Lesion leakage"):
        validate_splits(split_data)


@pytest.mark.parametrize(
    ("number_of_folds", "validation_folds", "test_folds"),
    [(5, 0, 1), (5, 1, 0), (4, 2, 2)],
)
def test_invalid_split_config_fails(
    number_of_folds: int, validation_folds: int, test_folds: int
) -> None:
    with pytest.raises(ValueError):
        SplitConfig(
            number_of_folds=number_of_folds,
            validation_folds=validation_folds,
            test_folds=test_folds,
        )


def test_save_splits_writes_manifests_and_reports(
    metadata: pd.DataFrame,
    tmp_path: Path,
) -> None:
    config = SplitConfig(random_seed=3)
    split_data = create_lesion_aware_splits(metadata, config)
    output_directory = tmp_path / "splits"
    report_directory = tmp_path / "reports"

    save_splits(split_data, output_directory, report_directory, config)

    manifests = [pd.read_csv(output_directory / f"{name}.csv") for name in SPLIT_NAMES]

    for name, manifest in zip(SPLIT_NAMES, manifests, strict=True):
        assert list(manifest.columns) == SPLIT_COLUMNS
        assert set(manifest["split"]) == {name}

    assert sum(len(manifest) for manifest in manifests) == len(metadata)

    for report_name in ("split_summary", "class_distribution", "source_distribution"):
        assert (report_directory / f"{report_name}.csv").exists()

    summary = pd.read_csv(report_directory / "split_summary.csv")
    assert summary["split"].tolist() == list(SPLIT_NAMES)
    assert summary["image_count"].sum() == len(metadata)
    assert summary["lesion_count"].sum() == metadata["lesion_id"].nunique()

    saved_config = json.loads((report_directory / "split_config.json").read_text())
    assert saved_config["random_seed"] == 3
    assert saved_config["group_column"] == "lesion_id"
    assert saved_config["fold_assignment"]["test"] == [17, 18, 19]
