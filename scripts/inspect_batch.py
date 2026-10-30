"""Load one preprocessed HAM10000 batch and print its properties."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
import yaml
from torchvision.utils import make_grid, save_image

from medvision.data.constants import SPLIT_NAMES
from medvision.data.loaders import LoaderConfig, create_dataloaders
from medvision.data.transforms import IMAGENET_MEAN, IMAGENET_STD

DEFAULT_CONFIG_PATH = Path("configs/data.yaml")
DEFAULT_PREVIEW_DIRECTORY = Path("reports/preprocessing")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Inspect one preprocessed HAM10000 batch.")

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML file with dataset, split, preprocessing, and loader settings.",
    )

    parser.add_argument(
        "--split",
        choices=SPLIT_NAMES,
        default="train",
    )

    parser.add_argument(
        "--save-preview",
        action="store_true",
        help=f"Save a denormalized image grid under {DEFAULT_PREVIEW_DIRECTORY}.",
    )

    return parser.parse_args()


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization for display."""

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    return (images * std + mean).clamp(0, 1)


def main() -> None:
    """Load one batch and print a short report."""

    arguments = parse_arguments()

    with arguments.config.open(encoding="utf-8") as file:
        settings = yaml.safe_load(file)

    preprocessing = settings["preprocessing"]
    image_size = preprocessing["image_size"]
    loader_config = LoaderConfig(**settings["loader"])

    loader = create_dataloaders(
        manifest_directory=settings["splits"]["output_dir"],
        dataset_root=settings["dataset"]["root"],
        config=loader_config,
        image_size=image_size,
        resize_size=preprocessing["resize_size"],
        splits=(arguments.split,),
    )[arguments.split]
    dataset = loader.dataset

    batch = next(iter(loader))
    images = batch["image"]
    targets = batch["target"]

    expected_batch_size = min(loader_config.batch_size, len(dataset))
    expected_shape = (expected_batch_size, 3, image_size, image_size)

    assert tuple(images.shape) == expected_shape, f"{tuple(images.shape)} != {expected_shape}"
    assert images.dtype == torch.float32
    assert tuple(targets.shape) == (expected_batch_size,)
    assert torch.isfinite(images).all()

    print(f"Split: {arguments.split} ({len(dataset):,} images, {len(loader):,} batches)")
    print(f"Image batch: {images.shape} {images.dtype}")
    print(f"Value range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Targets: {targets.shape} {targets.dtype} {targets.tolist()}")
    print(f"Classes: {dict(sorted(Counter(batch['class_name']).items()))}")
    print(f"Sources: {dict(sorted(Counter(batch['source']).items()))}")
    print(f"First image IDs: {batch['image_id'][:4]}")
    print(f"First lesion IDs: {batch['lesion_id'][:4]}")

    if arguments.save_preview:
        DEFAULT_PREVIEW_DIRECTORY.mkdir(parents=True, exist_ok=True)
        preview_path = DEFAULT_PREVIEW_DIRECTORY / f"batch_preview_{arguments.split}.jpg"
        grid = make_grid(denormalize(images), nrow=4, padding=2)
        save_image(grid, preview_path)
        print(f"Preview saved to: {preview_path}")


if __name__ == "__main__":
    main()
