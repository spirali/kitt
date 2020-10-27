import math

import pytest

from kitt.dataloading import BatchGenerator, EagerGenerator


class TestGenerator(BatchGenerator):
    def __init__(self, dataset, batch_size: int):
        super().__init__(dataset, batch_size)

    def load_sample(self, index):
        return [1, 2, 3, 4], [[1, 2, 3], [3, 4, 5]]


def check_generator(generator, length):
    batch_size = generator.batch_size
    assert len(generator) == math.ceil(length / batch_size)

    remaining = length
    for (x, y) in generator:
        expected_size = min(remaining, batch_size)
        assert x.shape == (expected_size, 4)
        assert y.shape == (expected_size, 2, 3)
        remaining -= expected_size


@pytest.mark.parametrize("batch_size", range(1, 9))
def test_batch_size(batch_size):
    dataset = list(range(7))
    generator = TestGenerator(len(dataset), batch_size)
    check_generator(generator, len(dataset))


def test_eager_loading():
    batch_size = 2
    dataset = list(range(7))
    generator = EagerGenerator(TestGenerator(len(dataset), batch_size), 3)
    check_generator(generator, len(dataset))
