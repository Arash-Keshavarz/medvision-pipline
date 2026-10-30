"""Create lesion-aware HAM10000 train, validation, and test manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from medvision.data.splits import (
    SplitConfig,
    create_lesion_aware_splits,
    load_metadata,
    save_splits,
)

DEFAULT_CONFIG_PATH = Path("configs/data.yaml")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Create lesion-aware HAM10000 splits.")

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML file with dataset and split settings.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed from the configuration.",
    )

    return parser.parse_args()


def main() -> None:
    """Create and save the split manifests and reports."""

    arguments = parse_arguments()

    with arguments.config.open(encoding="utf-8") as file:
        settings = yaml.safe_load(file)

    split_settings = settings["splits"]
    dataset_root = Path(settings["dataset"]["root"])
    output_directory = Path(split_settings["output_dir"])
    report_directory = Path(split_settings["report_dir"])

    config = SplitConfig(
        random_seed=(
            arguments.seed if arguments.seed is not None else split_settings["random_seed"]
        ),
        number_of_folds=split_settings["number_of_folds"],
        validation_folds=split_settings["validation_folds"],
        test_folds=split_settings["test_folds"],
    )

    print(f"Dataset root: {dataset_root}")

    metadata = load_metadata(dataset_root)
    split_data = create_lesion_aware_splits(metadata, config)
    saved_config = save_splits(
        split_data=split_data,
        output_directory=output_directory,
        report_directory=report_directory,
        config=config,
    )

    print(f"Random seed: {config.random_seed}")

    for split_name, ratio in saved_config["actual_image_ratios"].items():
        lesion_ratio = saved_config["actual_lesion_ratios"][split_name]
        print(f"{split_name}: {ratio:.2%} of images, {lesion_ratio:.2%} of lesions")

    print(f"Manifests saved to: {output_directory}")
    print(f"Reports saved to: {report_directory}")


if __name__ == "__main__":
    main()
