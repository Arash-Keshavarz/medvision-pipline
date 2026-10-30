"""Seeding and device selection.

Seeding makes runs repeatable on the same machine, device, and library
versions. It does not guarantee bitwise-identical results across devices
(CPU, CUDA, MPS) or library versions, and GPU kernels may be
non-deterministic even with a fixed seed.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU and all accelerators)."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _is_available(device_type: str) -> bool:
    if device_type == "cpu":
        return True

    if device_type == "cuda":
        return torch.cuda.is_available()

    if device_type == "mps":
        return torch.backends.mps.is_available()

    return False


def select_device(requested: str = "auto") -> torch.device:
    """Select a device: CUDA, then MPS, then CPU for ``auto``.

    An explicitly requested device must be available.
    """

    if requested == "auto":
        for device_type in ("cuda", "mps"):
            if _is_available(device_type):
                return torch.device(device_type)

        return torch.device("cpu")

    if requested not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unknown device {requested!r}; expected auto, cpu, cuda, or mps.")

    if not _is_available(requested):
        raise RuntimeError(f"Requested device {requested!r} is not available.")

    return torch.device(requested)
