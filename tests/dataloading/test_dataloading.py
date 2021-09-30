import math

import numpy as np
import pytest

from kitt.dataloading import (
    BatchLoader,
    DataLoader,
    EagerLoader,
    ListDataLoader,
    MappingLoader,
    ZipLoader,
)
from kitt.dataloading.tf import KerasSequence


@pytest.mark.parametrize("batch_size", range(1, 9))
def test_batching(batch_size):
    loader = ListDataLoader(list(range(70)))
    generator = BatchLoader(loader, batch_size)

    length = len(loader)
    assert len(generator) == math.ceil(length / batch_size)

    remaining = length
    for items in generator:
        expected_size = min(remaining, batch_size)
        assert len(items) == expected_size
        for item in items:
            assert 0 <= item < 70
        remaining -= expected_size


def test_batching_no_shuffle():
    dataset = [i for i in range(100)]

    loader = ListDataLoader(dataset)
    loader = BatchLoader(loader, batch_size=2, shuffle=False)

    def check():
        for (index, batch) in enumerate(loader):
            assert batch[0] == index * 2
            assert batch[1] == index * 2 + 1

    check()
    loader.reset()
    check()


def test_eager_loading():
    load_count = 0

    class Loader(DataLoader):
        def __len__(self):
            return 5

        def __getitem__(self, index):
            nonlocal load_count
            load_count += 1
            return index

    loader = Loader()
    loader = EagerLoader(loader)
    list(loader)
    list(loader)
    assert load_count == 5


def test_mapping():
    items = [
        np.array([1, 2, 3]),
        np.array([50, 2, 13]),
        np.array([8, 3, 81]),
        np.array([10, 5, 3]),
        np.array([51, 4, 2]),
    ]

    loader = ListDataLoader(items)
    loader = MappingLoader(loader, map_fn=lambda x: x * 2)

    for (index, item) in enumerate(loader):
        assert (item == items[index] * 2).all()


def test_zip_iterate():
    items = [
        np.array([1, 2, 3]),
        np.array([50, 2, 13]),
        np.array([8, 3, 81]),
        np.array([10, 5, 3]),
        np.array([51, 4, 2]),
    ]

    a = ListDataLoader(items)
    b = list(range(len(items)))

    for (item, index) in ZipLoader(a, b):
        assert (item == items[index]).all()


def test_zip_length():
    loader = ZipLoader([1, 2, 3], ["a", "b"], [True])
    assert len(loader) == 1
    assert list(loader) == [(1, "a", True)]


def test_keras_sequence():
    items = [
        (np.array([1, 2, 3]), np.array([1])),
        (np.array([2, 2, 13]), np.array([2])),
        (np.array([3, 3, 81]), np.array([3])),
        (np.array([4, 5, 3]), np.array([4])),
    ]

    loader = ListDataLoader(items)
    loader = BatchLoader(loader, batch_size=2)
    sequence = KerasSequence(loader)

    for (xs, ys) in sequence:
        assert isinstance(xs, np.ndarray)
        assert isinstance(ys, np.ndarray)
        assert xs.shape == (2, 3)
        assert ys.shape == (2, 1)

        for (x, y) in zip(xs, ys):
            assert x[0] == y[0]
