import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageStat

TEST_DIR = Path(__file__).absolute().parent
ROOT_DIR = TEST_DIR.parent

sys.path.insert(0, str(ROOT_DIR))

from kitt.image.image.tf import load_image  # noqa


def in_ci() -> bool:
    return "CI" in os.environ


def data_path(path: str) -> str:
    return str(TEST_DIR / "data" / path)


def image_bless_enabled() -> bool:
    """
    If `BLESS_IMAGES` is in environment variables, `check_image_equality` will bless the reference
    image snapshots.
    """
    return "BLESS_IMAGES" in os.environ


def show_image_diff(reference: Image, image: Image):
    if not in_ci() and not image_bless_enabled():
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


def bless_image(image_rgb: np.ndarray, reference_path: Path):
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    image_br = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(reference_path), image_br)


def check_image_equality(image: np.ndarray, path: str, delta=0.01):
    """
    Checks that `image` is "close enough" to the image stored at the provided `path`.
    :param image: RGB OpenCV image.
    :param path: Path to a PNG image.
    :param delta: Maximum allowed difference between the sum of mean differences per element
    of the two images.
    """
    if in_ci():
        # It's difficult to correctly match the expected results on CI.
        return

    path = Path(path)
    if not path.is_file():
        if image_bless_enabled():
            logging.info(
                f"Reference image {path} does not exist, creating it with generated image"
            )
            bless_image(image, path)
            return
        else:
            raise Exception(f"Reference image {path} does not exist")

    generated_image = Image.fromarray(image).convert("RGB")
    reference = Image.open(path).convert("RGB")

    difference = ImageChops.difference(generated_image, reference)
    stat = ImageStat.Stat(difference)
    diff = sum(stat.mean)
    if diff > delta:
        if image_bless_enabled():
            bless_image(image, path)
            return
        show_image_diff(reference, generated_image)

        raise Exception(f"Image is not equal enough with reference at `{path}`, diff: {diff}")
