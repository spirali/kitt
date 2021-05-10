import math
from typing import Tuple

import numpy as np
from tensorflow.keras.utils import Sequence

from kitt.data import train_test_split


class DataLoader:
    """
    Loads individual items of a dataset.
    You must implement the `__len__` and `__getitem__` functions for the loader to work correctly.

    Usage:
    ```python
    class MyDataLoader(DataLoader):
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index: int):
            return load_item(self.paths[index])
    ```
    """

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __bool__(self):
        return len(self) > 0

    def reset(self):
        pass

    def __iter__(self):
        """Create a generator that iterate over the loader."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def split(self, test_ratio: float) -> Tuple["DataLoader", "DataLoader"]:
        """
        Splits the loader into two loaders, one with train data, and another one with test data.
        """
        raise NotImplementedError("This loader cannot be split")


class ListDataLoader(DataLoader):
    """
    Loader that simply returns all elements of the passed list.

    Usage:
    ```python
    items = [Image("a.jpg"), Image("b.jpg"), Image("c.jpg")]
    loader = ListDataloader(items)
    ```
    """

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]

    def split(self, test_ratio: float) -> Tuple["DataLoader", "DataLoader"]:
        train, test = train_test_split(list(self.items), test_ratio)
        return (ListDataLoader(train), ListDataLoader(test))


class LoaderWrapper(DataLoader):
    """
    Wraps a loader and delegates everything to it.
    Inherit from this class if you need to wrap another data loader.
    """

    def __init__(self, loader: DataLoader):
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, index: int):
        return self.loader[index]


class EagerLoader(LoaderWrapper):
    """
    Preloads all samples from the given loader and keeps them in memory.

    Usage:
    ```python
    class MyDataLoader(DataLoader):
        ...

        def __getitem__(self, index: int):
            return expensive_load(index)

    loader = MyDataLoader(...)
    loader = EagerLoader(loader)  # loads all items into memory
    ```
    """

    def __init__(self, loader: DataLoader):
        super().__init__(loader)
        self.items = tuple(loader)

    def __getitem__(self, index: int):
        return self.items[index]

    def split(self, test_ratio: float) -> Tuple["DataLoader", "DataLoader"]:
        train, test = train_test_split(list(self.items), test_ratio)
        return (ListDataLoader(train), ListDataLoader(test))


class MappingLoader(LoaderWrapper):
    """
    Loader that wraps another loader and applies a function to each item.

    Usage:
    ```python
    loader = BGRImageDataLoader(...)
    loader = MappingLoader(loader, lambda image: bgr_to_rgb(image))
    ```
    """

    def __init__(self, loader: DataLoader, map_fn):
        super().__init__(loader)
        self.map_fn = map_fn

    def __getitem__(self, index: int):
        return self.map_fn(self.loader[index])


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


class BatchLoader(DataLoader):
    """
    Loader that takes another loader and aggregates its items into batches.

    Usage:
    ```python
    loader = ListDataLoader([1, 2, 3, 4, 5])
    loader = BatchLoader(loader, batch_size=2)

    batches = list(loader)  # [[1, 3], [4, 5], [2]]
    ```
    """

    def __init__(self, loader: DataLoader, batch_size: int, seed=None, shuffle=True):
        self.indexer = BatchIndexer(len(loader), batch_size, seed, shuffle)
        self.loader = loader

    @property
    def batch_size(self):
        return self.indexer.batch_size

    def __len__(self):
        return len(self.indexer)

    def reset(self):
        self.indexer.reset()

    def __getitem__(self, index: int):
        batch = []

        start = index * self.batch_size
        end = start + self.batch_size
        for index in self.indexer.get_indices(start, end):
            item = self.loader[index]
            batch.append(item)

        return batch


class KerasSequence(Sequence):
    """
    Takes a loader that returns batches (lists) of tuples (x, y) and transforms them to a format
    expected by Keras.

    Usually you will want to wrap a `BatchLoader` by a `KerasSequence` before using it in Keras.

    Usage:
    ```python
    loader = MyDataLoader(...)
    loader = BatchLoader(loader, batch_size=8)
    sequence = KerasSequence(loader)

    keras_model.fit(sequence)
    ```
    """

    def __init__(self, loader: DataLoader):
        self.loader = loader
        if loader:
            item = loader[0]  # each item must be a list (batch)
            assert isinstance(item, (list, tuple))
            assert len(item[0]) == 2  # each batch item must be a pair

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, index: int):
        items = self.loader[index]
        xs = []
        ys = []
        for (x, y) in items:
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

    def on_epoch_end(self):
        self.loader.reset()
