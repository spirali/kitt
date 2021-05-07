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

    training = []
    for index in training_idx:
        training.append(items[index])

    test = []
    for index in test_idx:
        test.append(items[index])
    return training, test
