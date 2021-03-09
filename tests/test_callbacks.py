import os

import numpy as np
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense

from kitt.callbacks import ModelCheckpoint


def create_model():
    return Sequential([
        Input(shape=(1,)),
        Dense(4),
        Dense(1)
    ])


def metric_return_y(y_true, y_pred):
    return y_true


def test_checkpoint_best_n_max(tmpdir):
    checkpointer = ModelCheckpoint(os.path.join(tmpdir, "{epoch:02d}.{metric}.hdf5"),
                                   "metric",
                                   save_n_best=2)
    train(checkpointer, [1, 2, 1, 0, 5, 3, 2, 4, 6])
    assert sorted(os.listdir(tmpdir)) == [
        "05.5.0.hdf5",
        "09.6.0.hdf5",
        "final.hdf5"
    ]


def test_checkpoint_best_n_min(tmpdir):
    checkpointer = ModelCheckpoint(os.path.join(tmpdir, "{epoch:02d}.{metric}.hdf5"),
                                   "metric",
                                   mode="min",
                                   save_n_best=2)
    train(checkpointer, [-1, -2, -1, 0, -5, -3, -2, -4, -6])
    assert sorted(os.listdir(tmpdir)) == [
        "05.-5.0.hdf5",
        "09.-6.0.hdf5",
        "final.hdf5"
    ]


def test_checkpoint_every_n(tmpdir):
    checkpointer = ModelCheckpoint(os.path.join(tmpdir, "{epoch:02d}.{metric}.hdf5"),
                                   "metric",
                                   save_n_best=False,
                                   save_every_n_epochs=2)
    train(checkpointer, [1, 2, 3, 4])
    assert sorted(os.listdir(tmpdir)) == [
        "02.2.0.hdf5",
        "04.4.0.hdf5",
        "final.hdf5"
    ]


def test_checkpoint_every_n_best_n(tmpdir):
    checkpointer = ModelCheckpoint(os.path.join(tmpdir, "{epoch:02d}.{metric}.hdf5"),
                                   "metric",
                                   save_n_best=2,
                                   save_every_n_epochs=2)
    train(checkpointer, [1, 2, 3, 4, 5, 6])
    assert sorted(os.listdir(tmpdir)) == [
        "02.2.0.hdf5",
        "04.4.0.hdf5",
        "05.5.0.hdf5",
        "06.6.0.hdf5",
        "final.hdf5"
    ]


def test_checkpoint_save_final(tmpdir):
    checkpointer = ModelCheckpoint(os.path.join(tmpdir, "{epoch:02d}.{metric}.hdf5"),
                                   "metric", save_n_best=False)
    train(checkpointer, [1, 2, 3])
    assert os.listdir(tmpdir) == ["final.hdf5"]


def train(checkpointer, y_values):
    metric = metric_return_y
    metric.__name__ = "metric"

    def generator():
        for item in y_values:
            yield np.array([item]), np.array([item])

    model = create_model()
    model.compile(optimizer="adam", loss="mse", metrics=[metric])
    model.fit(
        generator(),
        steps_per_epoch=1,
        shuffle=False,
        epochs=len(y_values),
        callbacks=[checkpointer],
        workers=0
    )
