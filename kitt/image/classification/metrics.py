import tensorflow as tf
from tensorflow.keras.metrics import binary_accuracy

from . import TFMetric


def multilabel_binary_accuracy_metric(threshold=0.5) -> TFMetric:
    """
    Creates a metric function that processes the output of a multi-label classification network
    while treating it like a binary classification output.

    If the value of any class in GT/prediction is larger or equal to `threshold`, the whole sample
    will be considered to be positive. If not, it will be considered to be negative.
    Binary accuracy is then used to evaluate these postprocessed positive/negative samples.
    """

    def multilabel_bin_acc(gt, prediction):
        gt_any_true = tf.cast(tf.reduce_any(gt >= threshold, axis=1), tf.int32)
        pred_any_true = tf.cast(
            tf.reduce_any(prediction >= threshold, axis=1), tf.int32
        )
        return binary_accuracy(gt_any_true, pred_any_true)

    return multilabel_bin_acc
