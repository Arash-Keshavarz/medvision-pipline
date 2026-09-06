"""Run the MedVision dataset audit"""

from __future__ import annotations

import argparse
from pathlib import Path

from medvision.data.audit import (
    audit_dataset,
    create_summary,
    save_audit_results,
    validate_structure,
)


DEFAULT_DATASET_ROOT = Path(
    "data/raw/skin-cancer-isic/Skin cancer ISIC The International Skin Imaging Collaboration"
)

DEFAULT_OUTPUT_DIRECTORY = Path("reports/dataset_audit")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Audit the Skin Cancer ISIC dataset.")

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Directory containing the Train and Test folders.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory in which audit results will be stored.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the dataset audit."""

    arguments = parse_arguments()

    print(f"Auditing dataset: {arguments.data_root}")

    inventory = audit_dataset(arguments.data_root)
    summary = create_summary(inventory)
    structure = validate_structure(inventory)

    save_audit_results(
        inventory=inventory,
        output_directory=arguments.output_dir,
    )

    print(f"Total images: {summary['total_images']}")
    print(f"Readable images: {summary['readable_images']}")
    print(f"Unreadable images: {summary['unreadable_images']}")
    print(f"Discovered splits: {structure['discovered_splits']}")
    print(f"Discovered classes: {len(structure['discovered_classes'])}")
    print(f"Results saved to: {arguments.output_dir}")


if __name__ == "__main__":
    main()
