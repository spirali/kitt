import itertools

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
