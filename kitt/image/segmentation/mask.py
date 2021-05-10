import itertools
from typing import List

import cv2
import numpy as np
import seaborn as sns

DEFAULT_PALETTE = sns.color_palette("pastel", 100)


def color_bitmap_masks(mask, palette=DEFAULT_PALETTE):
    """
    Colors bitmaps using a default palette.
    Expects grayscale uint8 images.

    Returns RGB uint8 images.
    """
    colored = []
    for (color, bitmap) in zip(itertools.cycle(palette), mask):
        normalized = bitmap.astype(np.float32) / 255
        colorful = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
        img = cv2.multiply(colorful, np.array([color]))
        img = img * 255
        img = img.astype(np.uint8)
        colored.append(img)
    return colored


def overlay_masks(background: np.ndarray, masks: List[np.ndarray], alpha=1.0):
    """
    Overlays colored masks over a background (only places where mask is nonzero are overlaid).
    Alpha specifies how visible should the masks be.
    """
    for mask in masks:
        indices = np.where(np.any(mask > 0, axis=2))
        if indices[0].size:
            background[indices] = cv2.addWeighted(
                background[indices], 1 - alpha, mask[indices], alpha, 0
            ).squeeze()
    return background


def binarize_mask(mask: np.ndarray, threshold=0.5) -> np.ndarray:
    """
    Binarize a (possibly multi-channel) mask with the given threshold.

    Values lower than `threshold` will be set to zero, values higher will be set to 1.
    """
    assert np.issubdtype(mask.dtype, np.floating)
    _, result = cv2.threshold(mask, threshold, 1, cv2.THRESH_BINARY)
    if result.shape != mask.shape:
        result = result.reshape(mask.shape)
    return result
