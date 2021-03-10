import heapq
import logging
import os

from tensorflow.keras.callbacks import Callback


class ModelCheckpoint(Callback):
    def __init__(
        self,
        filepath: str,
        metric: str,
        mode: str = "max",
        save_every_n_epochs: int = None,
        save_n_best=1,
    ):
        """
        :param filepath: Filepath where to save the model.
        :param metric: Metric to observe.
        :param mode: "min" or "max"
        :param save_n_best: Save N best models,
        :param save_every_n_epochs: Save the model every N epochs
        """
        super().__init__()
        self.filepath = str(filepath)
        self.metric = metric
        self.save_n_best = save_n_best or 0
        self.save_every_n_epochs = save_every_n_epochs
        self.epochs_since_save = 0

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

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_save += 1

        metric_value = logs[self.metric]
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
        self.model.save(path, overwrite=True)

    def get_filepath(self, epoch, logs):
        return self.filepath.format(epoch=epoch, **logs)
