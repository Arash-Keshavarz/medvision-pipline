"""Dataset auditing utilities for HAM10000."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

REQUIRED_METADATA_COLUMNS = {
    "lesion_id",
    "image_id",
    "dx",
    "dx_type",
    "age",
    "sex",
    "localization",
    "dataset",
}

EXPECTED_CLASSES = {
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
}


def load_metadata(dataset_root: Path) -> pd.DataFrame:
    """Load and validate the HAM10000 metadata file."""

    metadata_path = dataset_root / "HAM10000_metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    metadata = pd.read_csv(metadata_path)

    missing_columns = REQUIRED_METADATA_COLUMNS - set(metadata.columns)
    if missing_columns:
        raise ValueError(
            "Metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return metadata


def discover_images(images_root: Path) -> list[Path]:
    """Discover supported image files recursively."""

    if not images_root.exists():
        raise FileNotFoundError(
            f"Image directory not found: {images_root}"
        )

    return sorted(
        path
        for path in images_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def inspect_image(
    image_path: Path,
    dataset_root: Path,
    metadata_row: pd.Series | None,
) -> dict[str, Any]:
    """Inspect one image and combine it with its metadata."""

    record: dict[str, Any] = {
        "image_id": image_path.stem,
        "relative_path": str(image_path.relative_to(dataset_root)),
        "extension": image_path.suffix.lower(),
        "readable": False,
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "pixel_count": None,
        "mode": None,
        "error": None,
        "has_metadata": metadata_row is not None,
        "lesion_id": None,
        "class_name": None,
        "dx_type": None,
        "age": None,
        "sex": None,
        "localization": None,
        "source": None,
    }

    if metadata_row is not None:
        record.update(
            {
                "lesion_id": metadata_row["lesion_id"],
                "class_name": metadata_row["dx"],
                "dx_type": metadata_row["dx_type"],
                "age": metadata_row["age"],
                "sex": metadata_row["sex"],
                "localization": metadata_row["localization"],
                "source": metadata_row["dataset"],
            }
        )

    try:
        with Image.open(image_path) as image:
            image.load()

            width, height = image.size

            record.update(
                {
                    "readable": True,
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / height,
                    "pixel_count": width * height,
                    "mode": image.mode,
                }
            )

    except (UnidentifiedImageError, OSError, ValueError) as error:
        record["error"] = str(error)

    return record


def audit_dataset(dataset_root: Path) -> pd.DataFrame:
    """Create a complete image inventory for HAM10000."""

    dataset_root = dataset_root.resolve()
    images_root = dataset_root / "images"

    metadata = load_metadata(dataset_root)

    if metadata["image_id"].duplicated().any():
        duplicate_ids = metadata.loc[
            metadata["image_id"].duplicated(keep=False),
            "image_id",
        ].tolist()

        raise ValueError(
            "Duplicate image_id values found in metadata: "
            f"{duplicate_ids[:10]}"
        )

    image_paths = discover_images(images_root)

    if not image_paths:
        raise ValueError(
            f"No supported image found under: {images_root}"
        )

    metadata_by_id = metadata.set_index("image_id")
    records: list[dict[str, Any]] = []

    for image_path in image_paths:
        image_id = image_path.stem

        metadata_row = (
            metadata_by_id.loc[image_id]
            if image_id in metadata_by_id.index
            else None
        )

        records.append(
            inspect_image(
                image_path=image_path,
                dataset_root=dataset_root,
                metadata_row=metadata_row,
            )
        )

    inventory = pd.DataFrame(records)

    metadata_ids = set(metadata["image_id"])
    discovered_ids = set(inventory["image_id"])

    inventory.attrs["metadata_rows"] = len(metadata)
    inventory.attrs["metadata_image_ids"] = len(metadata_ids)
    inventory.attrs["missing_image_ids"] = sorted(
        metadata_ids - discovered_ids
    )
    inventory.attrs["images_without_metadata"] = sorted(
        discovered_ids - metadata_ids
    )

    return inventory


def create_summary(inventory: pd.DataFrame) -> dict[str, Any]:
    """Create the main dataset summary."""

    readable = inventory[inventory["readable"]]

    class_counts = (
        inventory["class_name"]
        .dropna()
        .value_counts()
        .sort_index()
    )

    source_counts = (
        inventory["source"]
        .dropna()
        .value_counts()
        .sort_index()
    )

    mode_counts = (
        readable["mode"]
        .dropna()
        .value_counts()
        .sort_index()
    )

    extension_counts = (
        inventory["extension"]
        .value_counts()
        .sort_index()
    )

    return {
        "total_images": int(len(inventory)),
        "readable_images": int(inventory["readable"].sum()),
        "unreadable_images": int((~inventory["readable"]).sum()),
        "images_with_metadata": int(
            inventory["has_metadata"].sum()
        ),
        "images_without_metadata": int(
            (~inventory["has_metadata"]).sum()
        ),
        "metadata_without_images": len(
            inventory.attrs.get("missing_image_ids", [])
        ),
        "unique_image_ids": int(inventory["image_id"].nunique()),
        "unique_lesions": int(
            inventory["lesion_id"].dropna().nunique()
        ),
        "class_counts": {
            str(key): int(value)
            for key, value in class_counts.items()
        },
        "source_counts": {
            str(key): int(value)
            for key, value in source_counts.items()
        },
        "image_modes": {
            str(key): int(value)
            for key, value in mode_counts.items()
        },
        "extensions": {
            str(key): int(value)
            for key, value in extension_counts.items()
        },
        "minimum_width": (
            int(readable["width"].min()) if not readable.empty else None
        ),
        "median_width": (
            float(readable["width"].median())
            if not readable.empty
            else None
        ),
        "maximum_width": (
            int(readable["width"].max()) if not readable.empty else None
        ),
        "minimum_height": (
            int(readable["height"].min())
            if not readable.empty
            else None
        ),
        "median_height": (
            float(readable["height"].median())
            if not readable.empty
            else None
        ),
        "maximum_height": (
            int(readable["height"].max())
            if not readable.empty
            else None
        ),
    }


def validate_structure(inventory: pd.DataFrame) -> dict[str, Any]:
    """Validate HAM10000 classes, metadata, and image mappings."""

    discovered_classes = set(
        inventory["class_name"].dropna().unique()
    )

    missing_values = {
        column: int(inventory[column].isna().sum())
        for column in (
            "lesion_id",
            "class_name",
            "dx_type",
            "age",
            "sex",
            "localization",
            "source",
        )
    }

    return {
        "expected_classes": sorted(EXPECTED_CLASSES),
        "discovered_classes": sorted(discovered_classes),
        "missing_classes": sorted(
            EXPECTED_CLASSES - discovered_classes
        ),
        "unexpected_classes": sorted(
            discovered_classes - EXPECTED_CLASSES
        ),
        "missing_image_ids": inventory.attrs.get(
            "missing_image_ids",
            [],
        ),
        "images_without_metadata": inventory.attrs.get(
            "images_without_metadata",
            [],
        ),
        "duplicate_image_ids": int(
            inventory["image_id"].duplicated().sum()
        ),
        "missing_metadata_values": missing_values,
    }


def create_class_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    """Summarize images and lesions per diagnosis class."""

    return (
        inventory.dropna(subset=["class_name"])
        .groupby("class_name")
        .agg(
            image_count=("image_id", "count"),
            lesion_count=("lesion_id", "nunique"),
        )
        .reset_index()
        .sort_values("class_name")
    )


def create_source_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    """Summarize images, lesions, and classes per source."""

    return (
        inventory.dropna(subset=["source"])
        .groupby("source")
        .agg(
            image_count=("image_id", "count"),
            lesion_count=("lesion_id", "nunique"),
            class_count=("class_name", "nunique"),
        )
        .reset_index()
        .sort_values("source")
    )


def create_lesion_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    """Summarize the number of images belonging to each lesion."""

    return (
        inventory.dropna(subset=["lesion_id"])
        .groupby("lesion_id")
        .agg(
            image_count=("image_id", "count"),
            class_count=("class_name", "nunique"),
            source_count=("source", "nunique"),
            class_name=("class_name", "first"),
            source=("source", "first"),
        )
        .reset_index()
        .sort_values(
            ["image_count", "lesion_id"],
            ascending=[False, True],
        )
    )


def plot_class_distribution(
    inventory: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save the diagnosis-class distribution plot."""

    counts = (
        inventory["class_name"]
        .dropna()
        .value_counts()
        .sort_values(ascending=False)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))
    counts.plot(kind="bar")
    plt.title("HAM10000 Class Distribution")
    plt.xlabel("Diagnosis")
    plt.ylabel("Number of images")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_source_distribution(
    inventory: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save the source distribution plot."""

    counts = (
        inventory["source"]
        .dropna()
        .value_counts()
        .sort_values(ascending=False)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("HAM10000 Source Distribution")
    plt.xlabel("Source")
    plt.ylabel("Number of images")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_image_dimensions(
    inventory: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a scatter plot of readable image dimensions."""

    readable = inventory[inventory["readable"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    for source, group in readable.groupby("source"):
        plt.scatter(
            group["width"],
            group["height"],
            label=source,
            alpha=0.5,
            s=15,
        )

    plt.title("HAM10000 Image Dimension Distribution")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.legend(title="Source")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_audit_results(
    inventory: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Save audit tables, summaries, and plots."""

    output_directory.mkdir(parents=True, exist_ok=True)

    summary = create_summary(inventory)
    validation = validate_structure(inventory)

    report = {
        "summary": summary,
        "structure_validation": validation,
    }

    inventory.to_csv(
        output_directory / "image_inventory.csv",
        index=False,
    )

    create_class_summary(inventory).to_csv(
        output_directory / "class_summary.csv",
        index=False,
    )

    create_source_summary(inventory).to_csv(
        output_directory / "source_summary.csv",
        index=False,
    )

    create_lesion_summary(inventory).to_csv(
        output_directory / "lesion_summary.csv",
        index=False,
    )

    with (output_directory / "audit_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2)

    plot_class_distribution(
        inventory,
        output_directory / "class_distribution.png",
    )

    plot_source_distribution(
        inventory,
        output_directory / "source_distribution.png",
    )

    plot_image_dimensions(
        inventory,
        output_directory / "image_dimensions.png",
    )