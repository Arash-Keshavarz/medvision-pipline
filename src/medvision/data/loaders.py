"""DataLoader factories for HAM10000 split manifests."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from medvision.data.constants import SPLIT_NAMES
from medvision.data.dataset import HAM10000Dataset
from medvision.data.transforms import DEFAULT_IMAGE_SIZE, DEFAULT_RESIZE_SIZE, build_transforms


@dataclass(frozen=True)
class LoaderConfig:
    """DataLoader settings.

    The defaults suit local development on a laptop with limited memory.
    """

    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy in a DataLoader worker.

    PyTorch seeds each worker's torch generator from the loader generator;
    this propagates that seed to the other random number generators.
    """

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(
    dataset: Dataset,
    split: str,
    config: LoaderConfig | None = None,
    shuffle: bool | None = None,
) -> DataLoader:
    """Create a DataLoader for one split.

    Training data are shuffled by default; validation and test data keep
    manifest order. A dedicated seeded generator makes the shuffle order and
    worker seeds reproducible. The final incomplete batch is kept.
    """

    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLIT_NAMES}.")

    config = config or LoaderConfig()

    if shuffle is None:
        shuffle = split == "train"

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    uses_workers = config.num_workers > 0

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers and uses_workers,
        worker_init_fn=seed_worker if uses_workers else None,
        generator=generator,
        drop_last=False,
    )


def create_dataloaders(
    manifest_directory: Path | str,
    dataset_root: Path | str,
    config: LoaderConfig | None = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
    resize_size: int = DEFAULT_RESIZE_SIZE,
    splits: tuple[str, ...] = SPLIT_NAMES,
) -> dict[str, DataLoader]:
    """Create DataLoaders for the requested splits from ``<split>.csv`` manifests."""

    manifest_directory = Path(manifest_directory)
    loaders: dict[str, DataLoader] = {}

    for split in splits:
        dataset = HAM10000Dataset(
            manifest_path=manifest_directory / f"{split}.csv",
            dataset_root=dataset_root,
            transform=build_transforms(split, image_size=image_size, resize_size=resize_size),
            expected_split=split,
        )
        loaders[split] = create_dataloader(dataset, split, config)

    return loaders
