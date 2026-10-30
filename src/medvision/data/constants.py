"""Canonical HAM10000 class and split definitions.

This module is the single owner of the class order. Model output index ``i``
always corresponds to ``CLASS_NAMES[i]``; changing the order invalidates
trained checkpoints.
"""

from __future__ import annotations

from types import MappingProxyType

CLASS_NAMES: tuple[str, ...] = (
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
)

NUM_CLASSES = len(CLASS_NAMES)

CLASS_TO_INDEX = MappingProxyType({name: index for index, name in enumerate(CLASS_NAMES)})

INDEX_TO_CLASS = MappingProxyType(dict(enumerate(CLASS_NAMES)))

EXPECTED_CLASSES = frozenset(CLASS_NAMES)

SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")
