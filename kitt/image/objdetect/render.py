from typing import List

import cv2
import numpy as np

from ..image import get_image_size
from .annotation import AnnotatedBBox, AnnotationType


def render_annotations(image: np.ndarray, annotations: List[AnnotatedBBox]):
    """Renders bounding boxes onto an image, optionally with labels and probabilities."""
    color_map = {
        AnnotationType.GROUND_TRUTH: (255, 255, 255),
        AnnotationType.PREDICTION: (255, 255, 0),
    }

    for annotation in annotations:
        box = annotation.bbox.as_denormalized(*get_image_size(image)).to_int()
        cv2.rectangle(
            image, box.top_left, box.bottom_right, color_map[annotation.type], 4
        )

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
                (box.top_left[0] + 20, box.top_left[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                color_map[annotation.type],
                2,
            )  # line type
