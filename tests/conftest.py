import sys
from pathlib import Path

import numpy as np

TEST_DIR = Path(__file__).absolute().parent
ROOT_DIR = TEST_DIR.parent

sys.path.insert(0, str(ROOT_DIR))

from kitt.image.image.tf import load_image  # noqa


def data_path(path: str) -> str:
    return str(TEST_DIR / "data" / path)


def check_image_equality(image: np.ndarray, path: str, delta=3):
    from PIL import Image, ImageChops, ImageStat

    image = Image.fromarray(image)
    reference = Image.open(path).convert("RGB")

    difference = ImageChops.difference(image, reference)
    stat = ImageStat.Stat(difference)
    diff = sum(stat.mean)
    if diff > delta:
        raise Exception("Images are not equal enough")
