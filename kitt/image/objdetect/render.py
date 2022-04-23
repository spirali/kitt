from typing import Callable, List

import cv2
import numpy as np

from ..color import BGRUintColor
from ..image import get_image_size
from .annotation import AnnotatedBBox, AnnotationType

DEFAULT_COLOR_MAP = {
    AnnotationType.GROUND_TRUTH: (0, 0, 255),
    AnnotationType.PREDICTION: (255, 0, 0),
}


def color_annotation(annotation: AnnotatedBBox) -> BGRUintColor:
    return DEFAULT_COLOR_MAP[annotation.type]


def render_annotations(
    image: np.ndarray,
    annotations: List[AnnotatedBBox],
    color_fn: Callable[[AnnotatedBBox], BGRUintColor] = None,
):
    """Renders bounding boxes onto an image, optionally with labels and probabilities."""
    if color_fn is None:
        color_fn = color_annotation

    image_size = get_image_size(image)
    width = image_size[0]
    bbox_width = min(5, max(1, width // 256))
    font_scale = width / 1920
    offset_horizontal = int(20 * (width / 1920))
    offset_vertical = int(40 * (width / 1920))

    for annotation in annotations:
        box = annotation.bbox.as_denormalized(*image_size).to_int()
        color = color_fn(annotation)
        cv2.rectangle(image, box.top_left, box.bottom_right, color, bbox_width)

        label = annotation.class_name
        confidence = annotation.confidence

        text_fields = []
        if label:
            text_fields.append(f"{label}")
        if confidence:
            text_fields.append(f"{confidence:.2f}")
        if text_fields:
            cv2.putText(
                image,
                " ".join(text_fields),
                (
                    box.top_left[0] + offset_horizontal,
                    box.top_left[1] + offset_vertical,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,  # line type
            )
