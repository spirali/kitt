import math

import numpy as np
from tensorflow.keras.utils import Sequence


# Generic data loaders
class DataLoader:
    """
    Loads individual items of a dataset (without batching).
    """

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def __iter__(self):
        """Create a generator that iterate over the loader."""
        for item in (self[i] for i in range(len(self))):
            yield item


class ListDataLoader(DataLoader):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class LoaderWrapper(DataLoader):
    """Wraps a loader and delegates everything to it."""

    def __init__(self, loader: DataLoader):
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, item):
        return self.loader[item]


class EagerLoader(LoaderWrapper):
    """Preloads all samples from the given sequence and keeps them in memory"""

    def __init__(self, loader: DataLoader):
        super().__init__(loader)
        self.items = tuple(loader)

    def __getitem__(self, index):
        return self.items[index]


class MappingLoader(LoaderWrapper):
    """Loader that wraps another loader and applies a function to its items"""

    def __init__(self, loader: DataLoader, map_fn):
        super().__init__(loader)
        self.map_fn = map_fn

    def __getitem__(self, item):
        return self.map_fn(self.loader[item])


# Keras adapters
class BatchIndexer:
    def __init__(self, length: int, batch_size: int, seed=None, shuffle=True):
        assert batch_size > 0
        assert length > 0

        if not shuffle:
            assert seed is None

        self.batch_size = batch_size
        self.batch_count = int(math.ceil(length / self.batch_size))
        self.indices = np.arange(0, length)
        self.shuffle = shuffle

        if shuffle:
            self.random = np.random.RandomState(seed)
            self.random.shuffle(self.indices)

    def get_indices(self, start, end):
        return self.indices[start:end]

    def reset(self):
        if self.shuffle:
            self.random.shuffle(self.indices)

    def __len__(self):
        return self.batch_count


class BatchSequence(Sequence):
    """
    Sequence that takes a loader and aggregates it into batches.
    It does not construct numpy arrays!
    You have to wrap the resulting generator in NumpyBatchGenerator.
    """

    def __init__(self, loader: DataLoader, batch_size: int, seed=None, shuffle=True):
        self.indexer = BatchIndexer(len(loader), batch_size, seed, shuffle)
        self.loader = loader

    @property
    def batch_size(self):
        return self.indexer.batch_size

    def __len__(self):
        return len(self.indexer)

    def on_epoch_end(self):
        self.indexer.reset()

    def __getitem__(self, index):
        xs = []
        ys = []

        start = index * self.batch_size
        end = start + self.batch_size
        for index in self.indexer.get_indices(start, end):
            x, y = self.loader[index]
            xs.append(x)
            ys.append(y)

        return xs, ys


class ToNumpySequence(Sequence):
    """
    Takes a Keras sequence that returns tuple (batch_x, batch_y) and transforms them to np.array.
    """

    def __init__(self, sequence: Sequence):
        self.sequence = sequence

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        xs, ys = self.sequence[index]
        return np.array(xs), np.array(ys)

    def on_epoch_end(self):
        self.sequence.on_epoch_end()
