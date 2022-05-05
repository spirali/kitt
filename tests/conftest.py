import os
import sys
from pathlib import Path

import numpy as np

TEST_DIR = Path(__file__).absolute().parent
ROOT_DIR = TEST_DIR.parent

sys.path.insert(0, str(ROOT_DIR))

from kitt.image.image.tf import load_image  # noqa


def in_ci() -> bool:
    return "CI" in os.environ


def data_path(path: str) -> str:
    return str(TEST_DIR / "data" / path)


def check_image_equality(image: np.ndarray, path: str, delta=0.001):
    """
    Checks that `image` is "close enough" to the image stored at the provided `path`.
    :param image: RGB OpenCV image.
    :param path: Path to a PNG image.
    :param delta: Maximum allowed difference between the sum of mean differences per element
    of the two images.
    """
    from PIL import Image, ImageChops, ImageDraw, ImageStat

    image = Image.fromarray(image).convert("RGB")
    reference = Image.open(path).convert("RGB")

    difference = ImageChops.difference(image, reference)
    stat = ImageStat.Stat(difference)
    diff = sum(stat.mean)
    if diff > delta:
        if not in_ci():
            merged = Image.new(
                "RGB",
                (image.width + reference.width, max(image.height, reference.height)),
            )
            merged.paste(reference, (0, 0))
            merged.paste(image, (reference.width, 0))
            draw = ImageDraw.Draw(merged)
            draw.line(
                (reference.width, 0, reference.width, reference.height),
                fill=(255, 0, 0),
                width=2,
            )
            merged.show("Comparison (reference - image)")

        raise Exception(f"Image is not equal enough with reference at `{path}`")
