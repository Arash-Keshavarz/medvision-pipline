"""Tests for HAM10000 image transforms."""

from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from medvision.data.transforms import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_RESIZE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_ROTATION_DEGREES,
    RandomRotationWithoutBorder,
    build_evaluation_transforms,
    build_train_transforms,
    build_transforms,
    inscribed_crop_size,
)


@pytest.fixture
def image() -> Image.Image:
    """A HAM10000-sized image with a non-uniform pattern."""

    pattern = Image.linear_gradient("L").resize((600, 450))
    return Image.merge("RGB", (pattern, pattern.rotate(90), pattern.transpose(0)))


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_transforms_return_normalized_float_tensor(image: Image.Image, split: str) -> None:
    output = build_transforms(split)(image)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 224, 224)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_transforms_handle_small_images() -> None:
    small = Image.new("RGB", (60, 45), (200, 120, 90))

    assert build_train_transforms()(small).shape == (3, 224, 224)
    assert build_evaluation_transforms()(small).shape == (3, 224, 224)


def test_custom_image_size() -> None:
    small = Image.new("RGB", (600, 450), (200, 120, 90))

    assert build_transforms("train", image_size=128)(small).shape == (3, 128, 128)
    assert build_transforms("test", image_size=128, resize_size=146)(small).shape == (3, 128, 128)


def test_evaluation_transform_is_deterministic(image: Image.Image) -> None:
    transform = build_evaluation_transforms()

    first = transform(image)
    torch.manual_seed(123)
    second = transform(image)

    assert torch.equal(first, second)


def test_train_transform_is_reproducible_with_seed(image: Image.Image) -> None:
    transform = build_train_transforms()

    torch.manual_seed(0)
    first = transform(image)
    torch.manual_seed(0)
    second = transform(image)

    assert torch.equal(first, second)


def test_normalization_uses_imagenet_statistics() -> None:
    mean_colour = Image.new("RGB", (256, 256), (124, 116, 104))
    output = build_evaluation_transforms()(mean_colour)

    # (124, 116, 104) / 255 is within half a grey level of the ImageNet mean.
    assert output.abs().max() < 0.01


def test_default_evaluation_keeps_full_height_and_crops_sides() -> None:
    # 600x450 grey image with red top/bottom bands and blue left/right bands.
    image = Image.new("RGB", (600, 450), (128, 128, 128))
    image.paste((255, 0, 0), (0, 0, 600, 20))
    image.paste((255, 0, 0), (0, 430, 600, 450))
    image.paste((0, 0, 255), (0, 0, 60, 450))
    image.paste((0, 0, 255), (540, 0, 600, 450))

    output = build_evaluation_transforms()(image)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    pixels = (output * std + mean) * 255

    middle = DEFAULT_IMAGE_SIZE // 2

    # Top and bottom edges survive, so the full height is kept.
    assert pixels[:, 0, middle].tolist() == pytest.approx([255, 0, 0], abs=3)
    assert pixels[:, -1, middle].tolist() == pytest.approx([255, 0, 0], abs=3)
    # The left and right 10% of the width are cropped away.
    assert pixels[:, middle, 0].tolist() == pytest.approx([128, 128, 128], abs=3)
    assert pixels[:, middle, -1].tolist() == pytest.approx([128, 128, 128], abs=3)


def test_config_matches_transform_defaults() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "data.yaml"
    preprocessing = yaml.safe_load(config_path.read_text())["preprocessing"]

    assert preprocessing["image_size"] == DEFAULT_IMAGE_SIZE == 224
    assert preprocessing["resize_size"] == DEFAULT_RESIZE_SIZE == 224


def test_unknown_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown split"):
        build_transforms("val")


def test_resize_smaller_than_crop_is_rejected() -> None:
    with pytest.raises(ValueError, match="resize_size"):
        build_evaluation_transforms(image_size=224, resize_size=200)


def test_inscribed_crop_size() -> None:
    assert inscribed_crop_size(600, 450, 0) == (600, 450)
    assert inscribed_crop_size(600, 450, 10) == inscribed_crop_size(600, 450, -10)

    width, height = inscribed_crop_size(600, 450, MAX_ROTATION_DEGREES)

    assert 0.8 * 600 < width < 600
    assert width / height == pytest.approx(600 / 450, abs=0.01)


def test_rotation_leaves_no_fill_border() -> None:
    white = Image.new("RGB", (600, 450), (255, 255, 255))
    rotation = RandomRotationWithoutBorder(MAX_ROTATION_DEGREES)

    for seed in range(20):
        torch.manual_seed(seed)
        rotated = rotation(white)

        # Any black fill from the rotation would show up as a dark pixel.
        assert rotated.convert("L").getextrema()[0] >= 250
