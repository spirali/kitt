import math

import numpy as np
import pytest

from kitt.dataloading import (
    BatchSequence,
    DataLoader,
    EagerLoader,
    ListDataLoader,
    MappingLoader,
)


@pytest.mark.parametrize("batch_size", range(1, 9))
def test_batching(batch_size):
    loader = ListDataLoader([(np.zeros((3, 3)), np.zeros((4, 2))) for _ in range(7)])
    generator = BatchSequence(loader, batch_size)

    length = len(loader)
    assert len(generator) == math.ceil(length / batch_size)

    remaining = length
    for (x, y) in generator:
        expected_size = min(remaining, batch_size)
        assert np.array(x).shape == (expected_size, 3, 3)
        assert np.array(y).shape == (expected_size, 4, 2)
        remaining -= expected_size


def test_batching_no_shuffle():
    dataset = [(i, i + 1) for i in range(100)]

    loader = ListDataLoader(dataset)
    loader = BatchSequence(loader, batch_size=2, shuffle=False)

    def check():
        for (index, (x, y)) in enumerate(loader):
            assert x[0] == index * 2
            assert x[1] == index * 2 + 1
            assert y[0] == index * 2 + 1
            assert y[1] == index * 2 + 2

    check()
    loader.on_epoch_end()
    check()


def test_eager_loading():
    load_count = 0

    class Loader(DataLoader):
        def __len__(self):
            return 5

        def __getitem__(self, item):
            nonlocal load_count
            load_count += 1
            return item

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
