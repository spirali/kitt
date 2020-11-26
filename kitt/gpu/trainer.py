import contextlib
import logging

import tensorflow as tf


class NullStrategy:
    def __init__(self, num_gpus: int):
        self.num_replicas_in_sync = num_gpus

    def scope(self):
        @contextlib.contextmanager
        def null_scope():
            yield

        return null_scope()


class GPUTrainer:
    """
    Facilitates training on multiple GPUs on a single node.

    Usage:
    ```python
    trainer = GPUTrainer(num_gpus=2)

    with trainer:
        model = # build model

    model.fit(batch_size=trainer.batch_size(32))
    ```
    """

    def __init__(self, num_gpus: int = 1):
        self.strategy = NullStrategy(num_gpus)
        self.num_gpus = num_gpus
        self.scope = None

        if num_gpus > 1:
            self.strategy = tf.distribute.MirroredStrategy()
        if num_gpus > 0:
            gpus = tf.config.list_physical_devices("GPU")
            if len(gpus) < num_gpus:
                logging.warning(
                    f"Requested {num_gpus} GPU(s), but only {len(gpus)} are available"
                )

    def batch_size(self, batch_size):
        if self.num_gpus > 1:
            actual_gpu_count = self.strategy.num_replicas_in_sync
            assert actual_gpu_count > 0
            return batch_size * actual_gpu_count
        return batch_size

    def __enter__(self):
        assert self.scope is None
        self.scope = self.strategy.scope()
        self.scope.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scope.__exit__(exc_type, exc_val, exc_tb)
        self.scope = None
