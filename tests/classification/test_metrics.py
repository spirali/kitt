import numpy as np

from kitt.image.classification.metrics import multilabel_binary_accuracy_metric


def test_multilabel_binary_accuracy_negative():
    gt = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    pred = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert multilabel_binary_accuracy_metric(threshold=0.5)(gt, pred) == 1.0


def test_multilabel_binary_accuracy_positive():
    gt = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    pred = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    assert multilabel_binary_accuracy_metric(threshold=0.5)(gt, pred) == 1.0


def test_multilabel_binary_accuracy_threshold():
    gt = np.array(
        [
            [0.0, 0.0, 0.0, 0.2],
        ]
    )
    pred = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert multilabel_binary_accuracy_metric(threshold=0.5)(gt, pred) == 1.0
    assert multilabel_binary_accuracy_metric(threshold=0.1)(gt, pred) == 0.0


def test_multilabel_binary_accuracy_complex():
    gt = np.array(
        [
            [0.0, 0.0, 0.0, 0.6],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.5, 0.6, 1.0],
        ]
    )
    pred = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.6, 0.0, 0.0, 1.0],
        ]
    )
    assert multilabel_binary_accuracy_metric(threshold=0.5)(gt, pred) == 0.5
