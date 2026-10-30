"""Image transforms for HAM10000 training and evaluation.

All preprocessing is applied on the fly; no transformed copies of images are
saved. Colour is deliberately left untouched (no colour jitter) so that the
baseline preserves skin-tone and lesion-colour information.
"""

from __future__ import annotations

import math

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, v2
from torchvision.transforms.v2 import functional as F

from medvision.data.constants import SPLIT_NAMES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_IMAGE_SIZE = 224

# Evaluation resizes the shorter side to DEFAULT_RESIZE_SIZE before the
# centre crop. With 224 on 600x450 images, the full height is kept but about
# 25% of the original width is removed in total (12.5% from each side).
DEFAULT_RESIZE_SIZE = 224

# Conservative training augmentation: keep at least 70% of the (rotated) image
# area so that lesions are rarely cropped out.
TRAIN_CROP_SCALE = (0.7, 1.0)
TRAIN_CROP_RATIO = (3 / 4, 4 / 3)
MAX_ROTATION_DEGREES = 10.0

# Pixels removed beyond the exact inscribed rectangle, so that interpolated
# edge pixels never blend with the rotation fill colour.
_ROTATION_EDGE_MARGIN = 2


def inscribed_crop_size(width: int, height: int, angle_degrees: float) -> tuple[int, int]:
    """Return the largest same-aspect crop that fits inside a rotated image.

    A centred ``w x h`` rectangle fits inside a ``width x height`` image rotated
    by ``angle`` when its own bounding box, rotated by the same angle, fits
    inside the original image.
    """

    angle = math.radians(abs(angle_degrees) % 180)
    angle = min(angle, math.pi - angle)
    cosine, sine = math.cos(angle), math.sin(angle)

    scale = min(
        width / (width * cosine + height * sine),
        height / (width * sine + height * cosine),
    )

    return max(1, math.floor(width * scale)), max(1, math.floor(height * scale))


class RandomRotationWithoutBorder:
    """Rotate by a small random angle and crop away the empty corners.

    Plain rotation fills the corners with a constant colour, which creates
    artificial black wedges that a classifier can learn. Cropping to the
    largest inscribed rectangle keeps only real image content. The angle is
    drawn from PyTorch's random number generator, so it follows
    ``torch.manual_seed`` and DataLoader worker seeding.
    """

    def __init__(self, degrees: float = MAX_ROTATION_DEGREES) -> None:
        if degrees < 0:
            raise ValueError("degrees must be non-negative.")

        self.degrees = degrees

    def __call__(self, image: Image.Image) -> Image.Image:
        angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees))

        if angle == 0:
            return image

        width, height = image.size
        rotated = F.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
        crop_width, crop_height = inscribed_crop_size(width, height, angle)

        return F.center_crop(
            rotated,
            [
                max(1, crop_height - _ROTATION_EDGE_MARGIN),
                max(1, crop_width - _ROTATION_EDGE_MARGIN),
            ],
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(degrees={self.degrees})"


def _to_normalized_tensor() -> list[v2.Transform]:
    """Convert a PIL image to a normalized float32 tensor.

    ``ToImage`` + ``ToDtype(scale=True)`` is the torchvision v2 equivalent of
    the deprecated ``ToTensor``.
    """

    return [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


def build_train_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> v2.Compose:
    """Build the stochastic training transform."""

    return v2.Compose(
        [
            RandomRotationWithoutBorder(MAX_ROTATION_DEGREES),
            v2.RandomResizedCrop(
                image_size,
                scale=TRAIN_CROP_SCALE,
                ratio=TRAIN_CROP_RATIO,
                antialias=True,
            ),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            *_to_normalized_tensor(),
        ]
    )


def build_evaluation_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    resize_size: int = DEFAULT_RESIZE_SIZE,
) -> v2.Compose:
    """Build the deterministic validation and test transform.

    The shorter side is resized to ``resize_size`` and the centre
    ``image_size`` square is kept. For non-square images this crops the
    longer sides, so lesions that extend to the image edge can lose part of
    their border.
    """

    if resize_size < image_size:
        raise ValueError("resize_size must be at least image_size.")

    return v2.Compose(
        [
            v2.Resize(resize_size, antialias=True),
            v2.CenterCrop(image_size),
            *_to_normalized_tensor(),
        ]
    )


def build_transforms(
    split: str,
    image_size: int = DEFAULT_IMAGE_SIZE,
    resize_size: int = DEFAULT_RESIZE_SIZE,
) -> v2.Compose:
    """Build the transform for a named split."""

    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLIT_NAMES}.")

    if split == "train":
        return build_train_transforms(image_size)

    return build_evaluation_transforms(image_size, resize_size)
