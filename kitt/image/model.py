from typing import Callable, Tuple

import numpy as np


class SimpleImageModel:
    def __init__(self, image_size: Tuple[int, int],
                 preprocess_fn: Callable[[np.ndarray], np.ndarray]):
        self.image_size = image_size
        self.preprocess_fn = preprocess_fn

    def build_network(self):
        raise NotImplementedError()
