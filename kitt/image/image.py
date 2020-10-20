from typing import Tuple, Union

import cv2
import numpy as np


def load_image_rgb(
    path: str, target_size: Union[None, Tuple[int, int]] = None
) -> np.ndarray:
    """Load an RGB image from the given path, optionally resizing it."""
    from tensorflow.keras.preprocessing.image import load_img

    pil = load_img(path, target_size=target_size)
    return np.array(pil)


def create_empty_image_rgb(size: Tuple[int, int]) -> np.ndarray:
    return np.zeros((*size, 3), dtype=np.uint8)


def resize_if_needed(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resizes an image if it is needed.
    Assumes image shape (height, width, channels)."""
    size = image.shape[:2]
    if size == target_size:
        return image
    return cv2.resize(image, target_size)
