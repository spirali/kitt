from typing import Iterable, Tuple

import cv2
import numpy as np


def polygons_to_binary_mask(
    shape: Tuple[int, int], polygons: Iterable[Iterable[Tuple[int, int]]]
) -> np.ndarray:
    """
    Creates a binary mask from a list of polygons.

    Points inside the polygon will have value 1, polygons outside will have value 0.
    """
    mask = np.zeros(shape, dtype=np.float)
    for polygon in polygons:
        cv2.fillPoly(mask, np.int32([polygon]), 1.0)
    return mask
