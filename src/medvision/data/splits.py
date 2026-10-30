"""Leakage-safe splitting utilities for HAM10000."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from medvision.data.constants import EXPECTED_CLASSES, SPLIT_NAMES

SPLIT_COLUMNS = [
    "image_id",
    "lesion_id",
    "class_name",
    "source",
    "relative_path",
    "split",
]

REQUIRED_METADATA_COLUMNS = {
    "image_id",
    "lesion_id",
    "dx",
    "dataset",
}


@dataclass(frozen=True)
class SplitConfig:
    """Fold-based split configuration.

    The data are divided into ``number_of_folds`` lesion-grouped folds.
    The last ``test_folds`` folds form the test split, the preceding
    ``validation_folds`` form the validation split, and the rest form
    the training split. Ratios are therefore approximate.
    """

    random_seed: int = 42
    number_of_folds: int = 20
    validation_folds: int = 3
    test_folds: int = 3

    def __post_init__(self) -> None:
        if self.validation_folds < 1 or self.test_folds < 1:
            raise ValueError("validation_folds and test_folds must be at least 1.")

        if self.train_folds < 1:
            raise ValueError("number_of_folds must leave at least one training fold.")

    @property
    def train_folds(self) -> int:
        """Number of folds assigned to the training split."""

        return self.number_of_folds - self.validation_folds - self.test_folds


def load_metadata(dataset_root: Path) -> pd.DataFrame:
    """Load HAM10000 metadata and add relative image paths."""

    metadata_path = dataset_root / "HAM10000_metadata.csv"
    image_root = dataset_root / "images"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    if not image_root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_root}")

    metadata = pd.read_csv(metadata_path)

    missing_columns = REQUIRED_METADATA_COLUMNS - set(metadata.columns)

    if missing_columns:
        raise ValueError(f"Missing metadata columns: {sorted(missing_columns)}")

    metadata = metadata.rename(
        columns={
            "dx": "class_name",
            "dataset": "source",
        }
    )

    validate_metadata(metadata)

    metadata["relative_path"] = metadata["image_id"].map(lambda image_id: f"images/{image_id}.jpg")

    missing_images = [
        relative_path
        for relative_path in metadata["relative_path"]
        if not (dataset_root / relative_path).exists()
    ]

    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} metadata images are missing, e.g. {missing_images[:5]}"
        )

    return metadata[SPLIT_COLUMNS[:-1]]


def validate_metadata(metadata: pd.DataFrame) -> None:
    """Validate metadata before splitting."""

    required_columns = {"image_id", "lesion_id", "class_name", "source"}
    missing_columns = required_columns - set(metadata.columns)

    if missing_columns:
        raise ValueError(f"Missing metadata columns: {sorted(missing_columns)}")

    null_counts = metadata[sorted(required_columns)].isna().sum()
    null_columns = null_counts[null_counts > 0]

    if not null_columns.empty:
        raise ValueError(f"Metadata contains missing values: {null_columns.to_dict()}")

    if metadata["image_id"].duplicated().any():
        duplicated_ids = metadata.loc[metadata["image_id"].duplicated(), "image_id"]
        raise ValueError(f"Duplicate image IDs: {duplicated_ids.tolist()[:10]}")

    unexpected_classes = set(metadata["class_name"]) - EXPECTED_CLASSES

    if unexpected_classes:
        raise ValueError(f"Unexpected classes: {sorted(unexpected_classes)}")

    class_counts_per_lesion = metadata.groupby("lesion_id")["class_name"].nunique()
    conflicting_lesions = class_counts_per_lesion[class_counts_per_lesion > 1]

    if not conflicting_lesions.empty:
        raise ValueError(
            "Some lesion IDs have multiple diagnosis labels: "
            f"{conflicting_lesions.index.tolist()[:10]}"
        )


def create_lesion_aware_splits(
    metadata: pd.DataFrame,
    config: SplitConfig | None = None,
) -> pd.DataFrame:
    """Create approximately stratified splits grouped by lesion ID."""

    config = config or SplitConfig()

    validate_metadata(metadata)

    # Sort and reset so that the output is independent of input order and
    # scikit-learn's positional indices can be used directly.
    result = metadata.sort_values("image_id").reset_index(drop=True)

    splitter = StratifiedGroupKFold(
        n_splits=config.number_of_folds,
        shuffle=True,
        random_state=config.random_seed,
    )

    folds = np.full(len(result), -1, dtype=int)

    for fold_index, (_, held_out_positions) in enumerate(
        splitter.split(
            result,
            y=result["class_name"],
            groups=result["lesion_id"],
        )
    ):
        folds[held_out_positions] = fold_index

    if (folds < 0).any():
        raise RuntimeError("At least one image was not assigned to a fold.")

    validation_start = config.train_folds
    test_start = config.train_folds + config.validation_folds

    result["fold"] = folds
    result["split"] = np.select(
        [folds < validation_start, folds < test_start],
        ["train", "validation"],
        default="test",
    )

    validate_splits(result)

    return result


def validate_splits(split_data: pd.DataFrame) -> None:
    """Validate split coverage, uniqueness, and lesion separation."""

    discovered_splits = set(split_data["split"])

    if discovered_splits != set(SPLIT_NAMES):
        raise ValueError(f"Unexpected splits: {sorted(discovered_splits)}")

    if split_data["image_id"].duplicated().any():
        raise ValueError("An image ID appears more than once.")

    lesion_split_counts = split_data.groupby("lesion_id")["split"].nunique()
    leaking_lesions = lesion_split_counts[lesion_split_counts > 1]

    if not leaking_lesions.empty:
        raise ValueError(f"Lesion leakage detected for {len(leaking_lesions)} lesions.")

    expected_classes = set(split_data["class_name"])

    for split_name, group in split_data.groupby("split"):
        missing_classes = expected_classes - set(group["class_name"])

        if missing_classes:
            raise ValueError(f"{split_name} is missing classes: {sorted(missing_classes)}")


def _distribution(split_data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Count images and lesions per split and value of ``column``."""

    distribution = (
        split_data.groupby(["split", column])
        .agg(
            image_count=("image_id", "count"),
            lesion_count=("lesion_id", "nunique"),
        )
        .reset_index()
    )

    split_totals = distribution.groupby("split")["image_count"].transform("sum")
    distribution["image_percentage_in_split"] = (
        100 * distribution["image_count"] / split_totals
    ).round(2)

    return _order_splits(distribution, [column])


def _order_splits(report: pd.DataFrame, extra_sort_columns: list[str]) -> pd.DataFrame:
    """Sort a report in train, validation, test order."""

    order = {split_name: index for index, split_name in enumerate(SPLIT_NAMES)}

    return (
        report.assign(_order=report["split"].map(order))
        .sort_values(["_order", *extra_sort_columns])
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def create_split_reports(
    split_data: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Create split-level reports."""

    split_summary = (
        split_data.groupby("split")
        .agg(
            image_count=("image_id", "count"),
            lesion_count=("lesion_id", "nunique"),
            class_count=("class_name", "nunique"),
            source_count=("source", "nunique"),
        )
        .reset_index()
    )

    split_summary["image_percentage"] = (
        100 * split_summary["image_count"] / split_summary["image_count"].sum()
    ).round(2)
    split_summary["lesion_percentage"] = (
        100 * split_summary["lesion_count"] / split_summary["lesion_count"].sum()
    ).round(2)

    split_summary = _order_splits(split_summary, [])[
        [
            "split",
            "image_count",
            "image_percentage",
            "lesion_count",
            "lesion_percentage",
            "class_count",
            "source_count",
        ]
    ]

    return {
        "split_summary": split_summary,
        "class_distribution": _distribution(split_data, "class_name"),
        "source_distribution": _distribution(split_data, "source"),
    }


def save_splits(
    split_data: pd.DataFrame,
    output_directory: Path,
    report_directory: Path,
    config: SplitConfig,
) -> dict[str, Any]:
    """Save manifests and split reports, and return the saved configuration."""

    output_directory.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)

    for split_name in SPLIT_NAMES:
        manifest = split_data[split_data["split"] == split_name].sort_values("image_id")

        manifest[SPLIT_COLUMNS].to_csv(
            output_directory / f"{split_name}.csv",
            index=False,
        )

    reports = create_split_reports(split_data)

    for report_name, report in reports.items():
        report.to_csv(
            report_directory / f"{report_name}.csv",
            index=False,
        )

    summary = reports["split_summary"].set_index("split")
    number_of_folds = config.number_of_folds

    split_config = {
        "strategy": "StratifiedGroupKFold",
        "group_column": "lesion_id",
        "stratification_column": "class_name",
        **asdict(config),
        "train_folds": config.train_folds,
        "fold_assignment": {
            "train": list(range(0, config.train_folds)),
            "validation": list(range(config.train_folds, number_of_folds - config.test_folds)),
            "test": list(range(number_of_folds - config.test_folds, number_of_folds)),
        },
        "target_ratios": {
            "train": round(config.train_folds / number_of_folds, 4),
            "validation": round(config.validation_folds / number_of_folds, 4),
            "test": round(config.test_folds / number_of_folds, 4),
        },
        "actual_image_ratios": {
            split_name: round(float(summary.loc[split_name, "image_percentage"]) / 100, 4)
            for split_name in SPLIT_NAMES
        },
        "actual_lesion_ratios": {
            split_name: round(float(summary.loc[split_name, "lesion_percentage"]) / 100, 4)
            for split_name in SPLIT_NAMES
        },
        "note": "Ratios are approximate because whole lesions are assigned to one split.",
    }

    with (report_directory / "split_config.json").open("w", encoding="utf-8") as file:
        json.dump(split_config, file, indent=2)

    return split_config
