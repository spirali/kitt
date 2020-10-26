from typing import Callable, Tuple

import numpy as np

from ..model import ModelWrapper


class SimpleImageModel(ModelWrapper):
    def __init__(
        self,
        image_size: Tuple[int, int],
        preprocess_fn: Callable[[np.ndarray], np.ndarray],
    ):
        super().__init__()
        self.image_size = image_size
        self.preprocess_fn = preprocess_fn

    def build_network(self):
        raise NotImplementedError()
