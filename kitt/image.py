from typing import Tuple

import cv2
import numpy as np


def load_image_np(path: str, rgb=True) -> np.ndarray:
    image = cv2.imread(path)
    if rgb:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def create_image_np(size: Tuple[int, int]) -> np.ndarray:
    return np.zeros((*size, 3), dtype=np.uint8)
