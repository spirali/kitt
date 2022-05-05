import itertools

import cv2
import numpy as np
import pytest
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense

from kitt.callbacks.tf.confusion_matrix import (
    ConfusionMatrixCallback,
    draw_confusion_matrices,
)
from kitt.dataloading import BatchLoader, ListDataLoader
from kitt.dataloading.tf import KerasSequence
from kitt.image.plot import render_plt_to_cv

from ..conftest import check_image_equality, data_path


def create_model(n_classes: int):
    return Sequential(
        [
            Input(shape=(1,)),
            Dense(32, activation="relu"),
            Dense(n_classes, activation="sigmoid"),
        ]
    )


def alternating_zero_one():
    while True:
        yield 0
        yield 1


@pytest.mark.parametrize("class_count", (1, 4))
def test_cm_callback(tmpdir, class_count: int):
    loader = ListDataLoader(
        [
            (0.1, list(itertools.islice(alternating_zero_one(), class_count))),
            (-0.1, list(itertools.islice(alternating_zero_one(), class_count))),
            (1.5, list(itertools.islice(alternating_zero_one(), class_count))),
        ]
    )
    loader = BatchLoader(loader, batch_size=2)
    sequence = KerasSequence(loader)

    cm = ConfusionMatrixCallback(tmpdir / "logs", sequence, every_n_epochs=1)
    train(sequence, cm, n_classes=class_count)


def test_draw_confusion_matrix_single_class(tmpdir):
    cm = [[5, 2], [0, 1]]
    draw_confusion_matrices(np.array(cm))
    image = render_plt_to_cv()
    check_image_equality(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB), data_path("confusion-matrix/cm1.png")
    )


def test_draw_confusion_matrix_multiple_classes(tmpdir):
    cm = [[[5, 2], [0, 1]], [[2, 1], [4, 0]]]
    draw_confusion_matrices(np.array(cm))
    image = render_plt_to_cv()
    check_image_equality(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB), data_path("confusion-matrix/cm2.png")
    )


def test_draw_confusion_matrix_multiple_rows(tmpdir):
    cm = [[[5, 2], [0, 1]], [[2, 1], [4, 0]], [[2, 1], [1, 1]], [[3, 2], [3, 3]]]
    draw_confusion_matrices(np.array(cm), columns=2)
    image = render_plt_to_cv()
    check_image_equality(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB), data_path("confusion-matrix/cm3.png")
    )


def test_draw_confusion_matrix_large_values(tmpdir):
    cm = [[[128, 5321], [850, 10001]]]
    draw_confusion_matrices(np.array(cm))
    image = render_plt_to_cv()
    check_image_equality(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB), data_path("confusion-matrix/cm4.png")
    )


def train(sequence: KerasSequence, callback: ConfusionMatrixCallback, n_classes: int):
    model = create_model(n_classes=n_classes)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(
        sequence,
        shuffle=False,
        epochs=4,
        callbacks=[callback],
        workers=0,
    )
