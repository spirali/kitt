import sys
from pathlib import Path

import numpy as np

TEST_DIR = Path(__file__).absolute().parent
ROOT_DIR = TEST_DIR.parent

sys.path.insert(0, str(ROOT_DIR))

from kitt.image.image import load_image  # noqa


def data_path(path: str) -> str:
    return str(TEST_DIR / "data" / path)


def check_image_equality(image: np.ndarray, path: str, color_mode="rgb", delta=None):
    reference = load_image(data_path(path), color_mode=color_mode)

    if delta is None:
        assert (image == reference).all()
    else:
        assert np.allclose(image[:, :], reference[:, :], rtol=delta)
