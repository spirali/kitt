import heapq
import logging
import os
from typing import List

from tensorflow.keras.callbacks import Callback

from ...utils import get_extension


class ModelCheckpoint(Callback):
    def __init__(
        self,
        filepath: str,
        monitor: str,
        mode: str = "max",
        save_every_n_epochs: int = None,
        save_n_best=1,
        save_optimizer=False,
    ):
        """
        :param filepath: Filepath where to save the model. Can contain "epoch" and "<monitor>"
        formatting placeholders.
        :param monitor: What metric to observe.
        :param mode: One of {"min", "max"}. Whether to consider the monitored metric to improve
        if it gets lower or higher.
        :param save_n_best: Save last N best models.
        :param save_every_n_epochs: Save the model every N epochs.
        :param save_optimizer: Include optimizer state in the saved model checkpoint.
        """
        super().__init__()

        if os.path.isdir(filepath) or get_extension(filepath) != ".hdf5":
            raise Exception(
                "Please use a placeholder path ending with .hdf5 in `filepath`"
            )

        self.filepath = str(filepath)
        self.monitor = monitor
        self.save_n_best = save_n_best or 0
        self.save_every_n_epochs = save_every_n_epochs
        self.epochs_since_save = 0
        self.save_optimizer = save_optimizer

        assert self.save_every_n_epochs is None or self.save_every_n_epochs > 0
        assert self.save_n_best >= 0

        if mode == "max":
            self.metric_map_fn = lambda x: x
        elif mode == "min":
            self.metric_map_fn = lambda x: -x
        else:
            raise Exception(f"Unknown mode {mode}")

        # Invariants
        # self.best_queue[0] is the worst saved model
        # self.best_queue[-1] is the best saved model
        self.best_queue = []

    def saved_models(self) -> List[str]:
        return [item[2] for item in self.best_queue]

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_save += 1

        metric_value = logs[self.monitor]
        path = self.get_filepath(epoch + 1, logs=logs)
        saved = False

        if self.save_every_n_epochs:
            if self.epochs_since_save % self.save_every_n_epochs == 0:
                self.epochs_since_save = 0
                self.save_model(path)
                saved = True

        if self.save_n_best > 0 and self.is_better(metric_value):
            self.push_better(epoch, metric_value, path, saved)
            if not saved:
                self.save_model(path)

    def on_train_end(self, logs=None):
        directory = os.path.dirname(self.filepath)
        self.save_model(os.path.join(directory, "final.hdf5"))

    def is_better(self, metric_value: float):
        if len(self.best_queue) < self.save_n_best:
            return True
        value = self.metric_map_fn(metric_value)
        return value > self.best_queue[0][0]

    def push_better(self, epoch: int, metric_value: float, path: str, pin=False):
        value = self.metric_map_fn(metric_value)
        heapq.heappush(self.best_queue, (value, epoch, path, pin))
        if len(self.best_queue) > self.save_n_best:
            _, _, previous_path, is_pinned = heapq.heappop(self.best_queue)
            if not is_pinned:
                try:
                    os.unlink(previous_path)
                except IOError as e:
                    logging.error(
                        f"Could not remove previously stored model path {previous_path}: {e}"
                    )

    def save_model(self, path: str):
        self.model.save(path, overwrite=True, include_optimizer=self.save_optimizer)

    def get_filepath(self, epoch, logs):
        return self.filepath.format(epoch=epoch, **logs)
