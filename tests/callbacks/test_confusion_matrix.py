import itertools

import cv2
import numpy as np
import pytest
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense

from kitt.callbacks.tf.confusion_matrix import (
    ConfusionMatrixCallback,
    calculate_confusion_matrix,
    draw_confusion_matrices,
)
from kitt.dataloading import BatchLoader, EmptyLoader, ListDataLoader
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


def test_cm_callback_empty_loader(tmpdir):
    loader = ListDataLoader(
        [
            (0.1, list(itertools.islice(alternating_zero_one(), 2))),
            (-0.1, list(itertools.islice(alternating_zero_one(), 2))),
            (1.5, list(itertools.islice(alternating_zero_one(), 2))),
        ]
    )
    loader = BatchLoader(loader, batch_size=2)
    sequence = KerasSequence(loader)

    val_sequence = KerasSequence(BatchLoader(EmptyLoader(), batch_size=2))

    cm = ConfusionMatrixCallback(tmpdir / "logs", val_sequence, every_n_epochs=1)
    train(sequence, cm, n_classes=2)


@pytest.mark.parametrize("class_count", (1, 2))
@pytest.mark.parametrize("value", (0, 1))
def test_draw_confusion_matrix_same_values(tmpdir, class_count: int, value: int):
    y_true = np.array([[value] * class_count, [value] * class_count])
    y_pred = np.array([[value] * class_count, [value] * class_count])

    cm = calculate_confusion_matrix(y_true, y_pred)
    assert cm.shape[0] == class_count

    draw_confusion_matrices(cm)
    image = render_plt_to_cv()
    check_image_equality(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        data_path(f"confusion-matrix/cm-{class_count}c-all-{value}.png"),
    )


def test_draw_confusion_matrix_single_class(tmpdir):
    # 3 TP, 2 TN, 1 FP, 0 FN
    y_true = np.array([[1], [0], [0], [1], [1], [0]])
    y_pred = np.array([[1], [0], [0], [1], [1], [1]])

    cm = calculate_confusion_matrix(y_true, y_pred)
    assert cm.shape[0] == 1

    draw_confusion_matrices(cm)
    image = render_plt_to_cv()
    check_image_equality(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB), data_path("confusion-matrix/cm1.png")
    )


def test_draw_confusion_matrix_multiple_classes(tmpdir):
    # class 0: 3 TP, 2 TN, 1 FP, 0 FN
    # class 1: 1 TP, 1 TN, 3 FP, 1 FN
    # class 2: 0 TP, 2 TN, 2 FP, 2 FN
    y_true = np.array(
        [[1, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]]
    )
    y_pred = np.array(
        [[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1], [0, 1, 0], [1, 0, 0]]
    )

    cm = calculate_confusion_matrix(y_true, y_pred)
    assert cm.shape[0] == 3

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
