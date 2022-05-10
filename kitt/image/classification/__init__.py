from typing import Callable

import numpy as np

# (ground truth batch, prediction batch) -> metric batch/metric scalar
TFMetric = Callable[[np.ndarray, np.ndarray], np.ndarray]
