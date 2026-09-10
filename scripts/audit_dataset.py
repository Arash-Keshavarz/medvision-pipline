"""Run the HAM10000 dataset audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from medvision.data.audit import (
    audit_dataset,
    create_summary,
    save_audit_results,
    validate_structure,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Audit the HAM10000 dataset."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/ham10000"),
        help="Directory containing images/ and HAM10000_metadata.csv.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/ham10000_audit"),
        help="Directory in which audit results are saved.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the audit and print its main results."""

    arguments = parse_arguments()

    inventory = audit_dataset(arguments.dataset_root)
    summary = create_summary(inventory)
    validation = validate_structure(inventory)

    save_audit_results(
        inventory=inventory,
        output_directory=arguments.output_dir,
    )

    print(f"Total images: {summary['total_images']}")
    print(f"Readable images: {summary['readable_images']}")
    print(f"Unreadable images: {summary['unreadable_images']}")
    print(f"Unique lesions: {summary['unique_lesions']}")
    print(
        "Images without metadata: "
        f"{summary['images_without_metadata']}"
    )
    print(
        "Metadata rows without images: "
        f"{summary['metadata_without_images']}"
    )
    print(
        "Discovered classes: "
        f"{validation['discovered_classes']}"
    )
    print(
        "Missing classes: "
        f"{validation['missing_classes']}"
    )
    print(
        "Unexpected classes: "
        f"{validation['unexpected_classes']}"
    )
    print(f"Results saved to: {arguments.output_dir}")


if __name__ == "__main__":
    main()