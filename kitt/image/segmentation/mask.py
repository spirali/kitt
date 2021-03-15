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
        indices = np.where(mask > 0)
        if all(np.any(i) for i in indices):
            background[indices] = cv2.addWeighted(background[indices], 1 - alpha, mask[indices],
                                                  alpha, 0).squeeze()
    return background
