import logging
import math
from typing import Iterable, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage

ImageSize = Tuple[int, int]


def create_empty_image_rgb(size: ImageSize) -> np.ndarray:
    return np.zeros((*size, 3), dtype=np.uint8)


def resize_if_needed(
    image: np.ndarray,
    target_size: ImageSize,
    keep_aspect_ratio=False,
    pad_color=(0, 0, 0),
) -> np.ndarray:
    """Resizes an image if it is needed.

    :param image: (height, width, channels)
    :param target_size: (width, height)
    :param keep_aspect_ratio: True/False
    :param pad_color: (R, G, B)/G"""

    size = image.shape[:2]
    if size == target_size[::-1]:
        return image

    h, w = image.shape[:2]
    sw, sh = target_size
    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interpolation = cv2.INTER_AREA
    else:  # stretching image
        logging.warning(
            f"Attempting to upsample from ({image.shape[1]},{image.shape[0]}) to {target_size}"
        )
        interpolation = cv2.INTER_CUBIC

    if keep_aspect_ratio:
        resized = resize_and_pad(image, target_size, pad_color, interpolation)
    else:
        resized = cv2.resize(image, target_size, interpolation)
    if resized.ndim < image.ndim:
        resized = np.expand_dims(resized, -1)
    return resized


def resize_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    """Resizes an image to the target width, while keeping aspect ratio.

    :param image: (height, width, channels)
    :param target_width: Target width
    """
    size = get_image_size(image)
    target_height = math.ceil((target_width / size[0]) * size[1])
    return resize_if_needed(image, target_size=(target_width, target_height))


def resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    """Resizes an image to the target width, while keeping aspect ratio.

    :param image: (height, width, channels)
    :param target_height: Target height
    """
    size = get_image_size(image)
    target_width = math.ceil((target_height / size[1]) * size[0])
    return resize_if_needed(image, target_size=(target_width, target_height))


def resize_and_pad(img, size, pad_color, interpolation):
    x, y = map(math.floor, size)
    new_w, new_h = x, y
    h, w = img.shape[:2]

    def round_aspect(number, key):
        return max(min(math.floor(number), math.ceil(number), key=key), 1)

    # preserve aspect ratio
    aspect = w / h
    if x / y >= aspect:
        x = round_aspect(y * aspect, key=lambda n: abs(aspect - n / y))
    else:
        y = round_aspect(x / aspect, key=lambda n: 0 if n == 0 else abs(aspect - x / n))
    pad_height = abs(y - new_h)
    pad_top = math.ceil(pad_height / 2)
    pad_bot = pad_height - pad_top

    pad_width = abs(x - new_w)
    pad_left = math.ceil(pad_width / 2)
    pad_right = pad_width - pad_left

    # set pad color
    if len(img.shape) == 3 and not isinstance(
        pad_color, (list, tuple, np.ndarray)
    ):  # color image but only one color provided
        pad_color = [pad_color] * 3

    # scale and pad
    scaled_img = cv2.resize(img, (x, y), interpolation=interpolation)
    scaled_img = cv2.copyMakeBorder(
        scaled_img,
        pad_top,
        pad_bot,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color,
    )
    return scaled_img


def display_image(image: np.ndarray, window="Kitt", wait=True):
    """
    Returns true if 'Q' was pressed.
    """
    cv2.imshow(window, rgb_to_bgr(image))
    if wait:
        return (cv2.waitKey() & 0xFF) == ord("q")
    return False


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def numpy_to_pillow(image: np.ndarray) -> PILImage.Image:
    return PILImage.fromarray(image)


def channels_first_to_last(arr: np.ndarray) -> np.ndarray:
    return np.transpose(arr, (1, 2, 0))


def channels_last_to_first(arr: np.ndarray) -> np.ndarray:
    return np.transpose(arr, (2, 0, 1))


def get_image_size(image: np.ndarray) -> ImageSize:
    return image.shape[:2][::-1]


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
