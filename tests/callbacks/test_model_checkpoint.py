import os
from pathlib import Path

import numpy as np
from tensorflow.keras import Input, Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense

from kitt.callbacks.tf.model_checkpoint import ModelCheckpoint


def create_model():
    return Sequential([Input(shape=(1,)), Dense(4), Dense(1)])


def metric_return_y(y_true, y_pred):
    return y_true


def test_checkpoint_best_n_max(tmpdir):
    checkpointer = ModelCheckpoint(
        os.path.join(tmpdir, "{epoch:02d}.{metric}.hdf5"),
        "metric",
        save_n_best=2,
        save_last_symlink=False,
    )
    train(checkpointer, [1, 2, 1, 0, 5, 3, 2, 4, 6])
    assert sorted(os.listdir(tmpdir)) == ["05.5.0.hdf5", "09.6.0.hdf5", "final.hdf5"]


def test_checkpoint_best_n_min(tmpdir):
    checkpointer = ModelCheckpoint(
        os.path.join(tmpdir, "{epoch:02d}.{metric}.hdf5"),
        "metric",
        mode="min",
        save_n_best=2,
        save_last_symlink=False,
    )
    train(checkpointer, [-1, -2, -1, 0, -5, -3, -2, -4, -6])
    assert sorted(os.listdir(tmpdir)) == ["05.-5.0.hdf5", "09.-6.0.hdf5", "final.hdf5"]


def test_checkpoint_every_n(tmpdir):
    checkpointer = ModelCheckpoint(
        os.path.join(tmpdir, "{epoch:02d}.{metric}.hdf5"),
        "metric",
        save_n_best=False,
        save_every_n_epochs=2,
        save_last_symlink=False,
    )
    train(checkpointer, [1, 2, 3, 4])
    assert sorted(os.listdir(tmpdir)) == ["02.2.0.hdf5", "04.4.0.hdf5", "final.hdf5"]


def test_checkpoint_every_n_best_n(tmpdir):
    checkpointer = ModelCheckpoint(
        os.path.join(tmpdir, "{epoch:02d}.{metric}.hdf5"),
        "metric",
        save_n_best=2,
        save_every_n_epochs=2,
        save_last_symlink=False,
    )
    train(checkpointer, [1, 2, 3, 4, 5, 6])
    assert sorted(os.listdir(tmpdir)) == [
        "02.2.0.hdf5",
        "04.4.0.hdf5",
        "05.5.0.hdf5",
        "06.6.0.hdf5",
        "final.hdf5",
    ]


def test_checkpoint_save_final(tmpdir):
    checkpointer = ModelCheckpoint(
        os.path.join(tmpdir, "{epoch:02d}.{metric}.hdf5"),
        "metric",
        save_n_best=False,
        save_last_symlink=False,
    )
    train(checkpointer, [1, 2, 3])
    assert os.listdir(tmpdir) == ["final.hdf5"]


def test_checkpoint_save_last_symlink(tmpdir):
    tmpdir = Path(tmpdir)

    class CheckCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch > 0:
                assert (tmpdir / "last-saved.hdf5").resolve() == (
                    tmpdir / f"{epoch:02}.{epoch}.0.hdf5"
                )

    checkpointer = ModelCheckpoint(
        os.path.join(tmpdir, "{epoch:02d}.{metric}.hdf5"),
        "metric",
        save_every_n_epochs=1,
        save_last_symlink=True,
    )
    train(checkpointer, [1, 2, 3], [CheckCallback()])
    assert (tmpdir / "last-saved.hdf5").resolve() == (tmpdir / "final.hdf5")


def train(checkpointer, y_values, callbacks=None):
    metric = metric_return_y
    metric.__name__ = "metric"

    def generator():
        for item in y_values:
            yield np.array([item]), np.array([item])

    if callbacks is None:
        callbacks = []
    callbacks.append(checkpointer)

    model = create_model()
    model.compile(optimizer="adam", loss="mse", metrics=[metric])
    model.fit(
        generator(),
        steps_per_epoch=1,
        shuffle=False,
        epochs=len(y_values),
        callbacks=callbacks,
        workers=0,
    )
