from typing import List

from keras_unet.metrics import dice_coef, iou, iou_thresholded


def get_default_metrics(classes: List[str]) -> List:
    metrics = [iou, iou_thresholded, dice_coef]
    if len(classes) > 1:
        metrics.extend(
            metric_iou_for_class(i, klass) for i, klass in enumerate(classes)
        )
    return metrics


def metric_iou_for_class(index: int, klass: str):
    def loss(y_true, y_pred):
        y_true_selected = y_true[..., index]
        y_pred_selected = y_pred[..., index]
        return iou(y_true_selected, y_pred_selected)

    loss.__name__ = f"iou_{klass}"
    return loss
