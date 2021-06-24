import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence

from . import DataLoader


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
