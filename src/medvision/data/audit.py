"""Dataset auditing utilities for the MedVision pipeline."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

SUPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

EXPECTED_SPLITS = {"Train", "Test"}

EXPECTED_CLASSES = {
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "vascular lesion",
}


def inspect_image(
    image_path: Path, dataset_root: Path, split: str, class_name: str
) -> dict[str, Any]:
    """Inspect one image and return its metadata."""

    record: dict[str, Any] = {
        "relative_path": image_path.relative_to(dataset_root).as_posix(),
        "split": split,
        "class_name": class_name,
        "extension": image_path.suffix.lower(),
        "file_size_bytes": image_path.stat().st_size,
        "readable": True,
        "width": None,
        "height": None,
        "mode": None,
        "error": None,
    }

    try:
        with Image.open(image_path) as img:
            record["width"], record["height"] = img.size
            record["mode"] = img.mode
            img.verify()  # Verify that the image is not corrupted

    except (UnidentifiedImageError, OSError, ValueError) as e:
        record["readable"] = False
        record["error"] = str(e)

    return record


def audit_dataset(dataset_root: Path) -> pd.DataFrame:
    """Inspect every supported image in the Dataset"""

    dataset_root = dataset_root.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root doesnt exist: {dataset_root}")

    records: list[dict[str, Any]] = []
    for split_directory in sorted(dataset_root.iterdir()):
        if not split_directory.is_dir():
            continue

        split = split_directory.name

        for class_directory in sorted(split_directory.iterdir()):
            if not class_directory.is_dir():
                continue

            class_name = class_directory.name

            for image_path in sorted(class_directory.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in SUPORTED_EXTENSIONS:
                    records.append(
                        inspect_image(
                            image_path=image_path,
                            dataset_root=dataset_root,
                            split=split,
                            class_name=class_name,
                        )
                    )
    if not records:
        raise ValueError(f"No supported image found under : {dataset_root}")

    return pd.DataFrame(records)


def validate_structure(inventory: pd.DataFrame) -> dict[str, Any]:
    """Validate discovered splits and classes."""

    discovered_splits = set(inventory["split"].unique())
    discovered_classes = set(inventory["class_name"].unique())

    return {
        "discovered_splits": sorted(discovered_splits),
        "missing_splits": sorted(EXPECTED_SPLITS - discovered_splits),
        "unexpected_splits": sorted(discovered_splits - EXPECTED_SPLITS),
        "discovered_classes": sorted(discovered_classes),
        "missing_classes": sorted(EXPECTED_CLASSES - discovered_classes),
        "unexpected_classes": sorted(discovered_classes - EXPECTED_CLASSES),
    }


def create_summary(inventory: pd.DataFrame) -> dict[str, Any]:
    """Create a JSON-serializable dataset summary."""

    readable = inventory[inventory["readable"]].copy()
    unreadable = inventory[~inventory["readable"]].copy()

    class_counts = inventory.groupby(["split", "class_name"]).size().sort_index()

    return {
        "total_images": int(len(inventory)),
        "readable_images": int(len(readable)),
        "unreadable_images": int(len(unreadable)),
        "class_counts": {
            f"{split}/{class_name}": int(count)
            for (split, class_name), count in class_counts.items()
        },
        "image_modes": {str(mode): int(count) for mode, count in Counter(readable["mode"]).items()},
        "minimum_width": int(readable["width"].min()),
        "maximum_width": int(readable["width"].max()),
        "minimum_height": int(readable["height"].min()),
        "maximum_height": int(readable["height"].max()),
        "extensions": {
            str(extension): int(count)
            for extension, count in Counter(inventory["extension"]).items()
        },
    }


def create_dimension_summary(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize image dimensions separately for every split."""

    readable = inventory[inventory["readable"]].copy()

    readable["pixel_count"] = readable["width"] * readable["height"]
    readable["aspect_ratio"] = readable["width"] / readable["height"]

    return (
        readable.groupby("split")
        .agg(
            image_count=("relative_path", "count"),
            minimum_width=("width", "min"),
            median_width=("width", "median"),
            maximum_width=("width", "max"),
            minimum_height=("height", "min"),
            median_height=("height", "median"),
            maximum_height=("height", "max"),
            median_pixel_count=("pixel_count", "median"),
            median_aspect_ratio=("aspect_ratio", "median"),
        )
        .reset_index()
    )


def save_class_distribution(
    inventory: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a class-distribution chart."""

    counts = inventory.groupby(["class_name", "split"]).size().unstack(fill_value=0).sort_index()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    axis = counts.plot(
        kind="bar",
        figsize=(12, 6),
        color={"Train": "#2F75B5", "Test": "#F4A261"},
    )

    axis.set_title("Skin Cancer ISIC Class Distribution")
    axis.set_xlabel("Class")
    axis.set_ylabel("Number of Images")
    axis.legend(title="Split")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def create_class_balance_summary(
    inventory: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize training-set class imbalance."""

    training_counts = (
        inventory[inventory["split"] == "Train"].groupby("class_name").size().sort_values()
    )

    minimum_count = int(training_counts.min())
    maximum_count = int(training_counts.max())

    return {
        "minimum_class": str(training_counts.idxmin()),
        "minimum_count": minimum_count,
        "maximum_class": str(training_counts.idxmax()),
        "maximum_count": maximum_count,
        "imbalance_ratio": round(
            maximum_count / minimum_count,
            2,
        ),
        "counts": {str(class_name): int(count) for class_name, count in training_counts.items()},
    }


def save_dimension_distribution(
    inventory: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a scatter plot of image dimensions."""

    readable = inventory[inventory["readable"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    for split, group in readable.groupby("split"):
        plt.scatter(
            group["width"],
            group["height"],
            label=split,
            alpha=0.5,
            s=15,
        )

    plt.title("Image Dimension Distribution")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_audit_results(
    inventory: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Save inventory, summaries, and figures."""

    output_directory.mkdir(parents=True, exist_ok=True)

    summary = create_summary(inventory)
    structure = validate_structure(inventory)
    class_imbalance = create_class_balance_summary(inventory)
    dimension_summary = create_dimension_summary(inventory)

    inventory.to_csv(output_directory / "dataset_inventory.csv", index=False)

    class_summary = (
        inventory.groupby(["split", "class_name"]).size().rename("image_count").reset_index()
    )

    class_summary.to_csv(
        output_directory / "class_summary.csv",
        index=False,
    )

    report = {
        "summary": summary,
        "structure_validation": structure,
        "training_class_imbalance": class_imbalance,
    }

    with (output_directory / "dataset_audit.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2)

    save_class_distribution(
        inventory,
        output_directory / "class_distribution.png",
    )

    save_dimension_distribution(
        inventory,
        output_directory / "image_dimensions.png",
    )

    dimension_summary.to_csv(
        output_directory / "dimension_summary.csv",
        index=False,
    )
