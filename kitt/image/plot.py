import io

import cv2
import numpy as np
from matplotlib import pyplot as plt


def render_plt_to_cv() -> np.ndarray:
    """
    Renders current plt figure to a BGR OpenCV image.
    """
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    array = np.asarray(bytearray(buffer.getvalue()), dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)
