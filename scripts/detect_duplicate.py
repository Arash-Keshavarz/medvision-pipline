"""Detect exact and near-duplicate images."""

from __future__ import annotations

import argparse
from pathlib import Path

from medvision.data.duplicates import (
    build_hash_inventory,
    create_duplicate_summary,
    find_exact_duplicate_groups,
    find_near_duplicate_pairs,
    save_duplicate_results,
)

DEFAULT_DATASET_ROOT = Path(
    "data/raw/skin-cancer-isic/Skin cancer ISIC The International Skin Imaging Collaboration"
)

DEFAULT_OUTPUT_DIRECTORY = Path("reports/duplicate_detection")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Detect duplicate images in the ISIC dataset.")

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )

    parser.add_argument(
        "--maximum-distance",
        type=int,
        default=6,
        help="Maximum perceptual-hash distance.",
    )

    return parser.parse_args()


def main() -> None:
    """Run duplicate detection."""

    arguments = parse_arguments()

    print("Building image hash inventory...")

    inventory = build_hash_inventory(arguments.data_root)

    print("Finding exact duplicates...")

    exact_groups = find_exact_duplicate_groups(inventory)

    print("Finding near-duplicate candidates...")

    near_pairs = find_near_duplicate_pairs(
        inventory=inventory,
        maximum_distance=arguments.maximum_distance,
    )

    save_duplicate_results(
        inventory=inventory,
        exact_groups=exact_groups,
        near_pairs=near_pairs,
        output_directory=arguments.output_dir,
    )

    summary = create_duplicate_summary(
        exact_groups=exact_groups,
        near_pairs=near_pairs,
    )

    print("\nDuplicate detection complete")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nResults saved to: {arguments.output_dir}")


if __name__ == "__main__":
    main()
