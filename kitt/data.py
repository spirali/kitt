from typing import List, Tuple, Union

import numpy as np


def train_test_split(items: List, test_ratio: float) -> Tuple[List, List]:
    """
    Splits the input list of `items` into a training and a test set.
    `test_ratio` should be a number in the interval [0.0, 1.0], which specifies the ratio of
    the test set size.

    :return: (train_items, test_items)
    """
    training_idx, test_idx = train_test_split_indices(len(items), test_ratio=test_ratio)

    training = []
    for index in training_idx:
        training.append(items[index])

    test = []
    for index in test_idx:
        test.append(items[index])
    return training, test


def train_test_split_indices(
    indices: Union[List, int], test_ratio: float
) -> Tuple[List[int], List[int]]:
    """
    Creates two lists of indices (train, test).
    `indices` can either be a list-like value from which the length is taken, or the length
    directly as an integer.
    `test_ratio` should be a number in the interval [0.0, 1.0], which specifies the ratio of
    the test set size.

    :return: (train_indices, test_indices)
    """
    length = int(indices) if isinstance(indices, int) else len(indices)
    indices = np.arange(length)
    np.random.shuffle(indices)

    start_index = int(length * test_ratio)
    training_idx, test_idx = indices[start_index:], indices[:start_index]
    return training_idx, test_idx


def to_onehot(value: int, dimension: int) -> np.array:
    """
    Creates a one-hot vector of size `dimension`, with `value` index set to 1.
    """
    vec = np.zeros(dimension, dtype=np.float32)
    vec[value] = 1
    return vec
