import logging
from typing import Iterable, Tuple, Union

import cv2
import numpy as np


def load_image(
    path: str,
    color_mode="rgb",
    target_size: Union[None, Tuple[int, int]] = None,
    normalize=False,
) -> np.ndarray:
    """Load an RGB image from the given path, optionally resizing it.

    :param path: Path to the image
    :param color_mode: "rgb", "bgr" or "grayscale"
    :param target_size: Target size of the image (width, height).
    :param normalize: Normalize values to [0.0, [1.0]
    """
    from tensorflow.keras.preprocessing.image import load_img

    pil_color_mode = color_mode
    if pil_color_mode == "bgr":
        pil_color_mode = "rgb"

    pil = load_img(path, color_mode=pil_color_mode, target_size=target_size)
    image = np.array(pil)
    if color_mode == "bgr":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if normalize:
        image = image / 255.0
    return image


def create_empty_image_rgb(size: Tuple[int, int]) -> np.ndarray:
    return np.zeros((*size, 3), dtype=np.uint8)


def resize_if_needed(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resizes an image if it is needed.

    :param image: (height, width, channels)
    :param target_size: (width, height)"""
    if np.any(target_size > image.shape[:2]):
        logging.warning(f"Attempting to upsample from {image.shape} to {target_size}")

    size = image.shape[:2]
    if size == target_size[::-1]:
        return image
    return cv2.resize(image, target_size)


def display_image(image: np.ndarray, window="Kitt", wait=True):
    cv2.imshow(window, rgb_to_bgr(image))
    if wait:
        cv2.waitKey()


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def create_image_grid(
    images: Iterable[np.ndarray], cols: int, border=False
) -> np.ndarray:
    """Concatenates images into a grid with `cols` columns.
    Draws borders between images if `border` is True.

    Assumes np.uint8 BGR images.
    """
    images = tuple(images)

    index = 0
    widths = []
    heights = []
    while index < len(images):
        widths.append(sum(image.shape[1] for image in images[index : index + cols]))
        heights.append(max(image.shape[0] for image in images[index : index + cols]))
        index += cols

    shape = (sum(heights), max(widths), 3)
    grid = np.zeros(shape, dtype=np.uint8)

    last_row = 0
    last_width = 0
    last_height = 0
    for index, image in enumerate(images):
        row = index // cols
        if row != last_row:
            last_height += heights[last_row]
            last_row = row
            last_width = 0

        height = image.shape[0]
        width = image.shape[1]

        if len(image.shape) < 3 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        grid[
            last_height : last_height + height, last_width : last_width + width
        ] = image
        if border:
            cv2.rectangle(
                grid,
                (last_width, last_height),
                (last_width + width, last_height + height),
                color=(0, 0, 255),
            )
        last_width += width

    return grid
