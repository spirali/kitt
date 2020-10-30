import math

import numpy as np
from tensorflow.keras.utils import Sequence


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


class BatchGenerator(Sequence):
    def __init__(self, length: int, batch_size: int, seed=None, shuffle=True):
        self.indexer = BatchIndexer(length, batch_size, seed, shuffle)

    @property
    def batch_size(self):
        return self.indexer.batch_size

    def load_sample(self, index):
        raise NotImplementedError()

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
            x, y = self.load_sample(index)
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)


class EagerGenerator(BatchGenerator):
    """Preloads all samples from the given sequence and keeps them in memory"""

    def __init__(self, sequence: Sequence, batch_size: int, seed=None, shuffle=True):
        self.samples = get_all_samples(sequence)
        super().__init__(len(self.samples), batch_size, seed, shuffle)

    def load_sample(self, index):
        return self.samples[index]


def get_all_samples(generator):
    samples = []
    for batch in generator:
        samples.extend(zip(*batch))
    return samples


class SequenceWrapper(Sequence):
    """Wraps a sequence and delegates everything to it."""

    def __init__(self, sequence: Sequence):
        self.sequence = sequence

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, item):
        return self.sequence[item]

    def on_epoch_end(self):
        self.sequence.on_epoch_end()


class MappingSequence(SequenceWrapper):
    """Sequence that wraps another sequence and applies a function to its batches"""

    def __init__(self, sequence: Sequence, map_x, map_y=None):
        super().__init__(sequence)
        self.map_x = map_x
        self.map_y = map_y if map_y else identity

    def __getitem__(self, item):
        xs, ys = self.sequence[item]
        return (self.map_x(xs), self.map_y(ys))


def identity(x):
    return x
