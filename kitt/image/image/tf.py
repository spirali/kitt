from typing import Union

import cv2
import numpy as np

from . import ImageSize


def load_image(
    path: str,
    color_mode="rgb",
    target_size: Union[None, ImageSize] = None,
    normalize=False,
) -> np.ndarray:
    """Load an RGB image from the given path, optionally resizing it.

    :param path: Path to the image
    :param color_mode: "rgb", "bgr" or "grayscale"
    :param target_size: Target size of the image (width, height).
    :param normalize: Normalize values to [0.0, 1.0]
    """
    from tensorflow.keras.preprocessing.image import load_img

    pil_color_mode = color_mode
    if pil_color_mode == "bgr":
        pil_color_mode = "rgb"

    if target_size is not None:
        target_size = target_size[::-1]

    pil = load_img(path, color_mode=pil_color_mode, target_size=target_size)
    image = np.array(pil)
    if color_mode == "bgr":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if normalize:
        image = image / 255.0
    return image
