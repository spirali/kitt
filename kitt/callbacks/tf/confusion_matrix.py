import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import multilabel_confusion_matrix
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Sequence

from ...files import GenericPath
from ...image.plot import render_plt_to_cv


def draw_confusion_matrices(confusion_matrices: np.ndarray, column_count=4):
    """
    Draws confusion matrices using pyplot.

    :param confusion_matrices: Numpy array containing one or more confusion matrices.
    Shape should be (2, 2) or (N, 2, 2).
    :param column_count: How many columns should the CM grid wrap to.
    """
    if confusion_matrices.ndim == 2:
        confusion_matrices = np.expand_dims(confusion_matrices, axis=0)

    df = pd.DataFrame(columns=["class", "gt", "prediction", "count"], dtype=int)

    for (class_index, cm) in enumerate(confusion_matrices):
        # rows are GTs, columns are predictions
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        frame = pd.DataFrame(
            {
                "class": [class_index] * 4,
                "gt": [0, 0, 1, 1],
                "prediction": [0, 1, 0, 1],
                "count": [tn, fp, fn, tp],
            }
        )
        df = pd.concat((df, frame))

    def draw(data, **kwargs):
        data = data.pivot("gt", "prediction", "count")
        sns.heatmap(data, annot=True)

    cols = min(column_count, len(confusion_matrices))
    row_count = int(math.ceil(len(confusion_matrices) / cols))

    g = sns.FacetGrid(df, col="class", col_wrap=cols)
    g.map_dataframe(draw)

    g.fig.set_figwidth(cols * 5)
    g.fig.set_figheight(row_count * 4)

    for ax in g.axes:
        ax.set_xlabel("Prediction")
        ax.set_ylabel("GT")


class ConfusionMatrixCallback(Callback):
    """
    Generates predictions the given sequence after each epoch.
    Assumes a classification model that returns multi-label data.
    The confusion matrices of the data is then rendered in a grid.
    """

    def __init__(
        self,
        log_dir: GenericPath,
        sequence: Sequence,
        every_n_epochs=10,
        round_threshold=0.5,
    ):
        """
        :param log_dir: Where to store TensorBoard output data.
        :param sequence: Input sequence which will be used to generate the confusion matrices.
        :param every_n_epochs: Only compute the output every N epochs.
        :param round_threshold: Threshold used for rounding the predictions to 0 or 1.
        """
        super().__init__()
        self.writer = tf.summary.create_file_writer(str(log_dir))
        self.sequence = sequence
        self.every_n_epochs = every_n_epochs
        self.round_threshold = round_threshold

    def on_epoch_end(self, epoch: int, *args, **kwargs):
        if (epoch % self.every_n_epochs) != 0:
            return

        predictions = []
        gts = []
        for (xs, ys) in self.sequence:
            preds = self.model.predict(xs)
            gts.extend(ys)
            predictions.extend(preds)
        predictions = (np.array(predictions) >= self.round_threshold).astype(np.float32)

        cms = multilabel_confusion_matrix(gts, predictions)

        def render():
            draw_confusion_matrices(cms, column_count=4)
            img = render_plt_to_cv()
            plt.close(plt.gcf())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return np.expand_dims(img, axis=0)

        with self.writer.as_default():
            tf.summary.image(
                "confusion_matrix",
                render(),
                step=epoch,
                description="Confusion matrices. Columns are predictions, rows are ground truths.",
            )
