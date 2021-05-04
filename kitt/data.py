from typing import List, Tuple

import numpy as np


def train_test_split(items: List, test_ratio: float) -> Tuple[List, List]:
    """
    Splits the input list of `items` into a training and a test set.
    `test_ratio` should be a number in the interval [0.0, 1.0], which specifies the ratio of
    the test set size.

    :return: (train_items, test_items)
    """
    length = len(items)
    indices = np.arange(length)
    np.random.shuffle(indices)

    start_index = int(length * test_ratio)
    training_idx, test_idx = indices[start_index:], indices[:start_index]

    items = np.array(items)
    training = list(items[training_idx])
    test = list(items[test_idx])
    return training, test
