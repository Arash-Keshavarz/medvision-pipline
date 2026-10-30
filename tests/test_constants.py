"""Tests for the canonical class mapping."""

import pytest

from medvision.data import audit, splits
from medvision.data.constants import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    EXPECTED_CLASSES,
    INDEX_TO_CLASS,
    NUM_CLASSES,
)


def test_class_order_is_stable() -> None:
    # Changing this order invalidates trained checkpoints.
    assert CLASS_NAMES == ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
    assert NUM_CLASSES == 7


def test_class_mapping_is_reversible() -> None:
    assert dict(CLASS_TO_INDEX) == {
        "akiec": 0,
        "bcc": 1,
        "bkl": 2,
        "df": 3,
        "mel": 4,
        "nv": 5,
        "vasc": 6,
    }

    for class_name, index in CLASS_TO_INDEX.items():
        assert INDEX_TO_CLASS[index] == class_name

    assert EXPECTED_CLASSES == set(CLASS_NAMES)


def test_class_mapping_is_read_only() -> None:
    with pytest.raises(TypeError):
        CLASS_TO_INDEX["new"] = 7  # type: ignore[index]


def test_audit_and_splits_share_the_constants() -> None:
    assert audit.EXPECTED_CLASSES is EXPECTED_CLASSES
    assert splits.EXPECTED_CLASSES is EXPECTED_CLASSES
