from typing import List, Tuple

import cv2
import numpy as np

from .annotation import AnnotatedImage, Annotation, BBox


def render_annotated_image(
    annotated: AnnotatedImage, color=(0, 0, 255), line_width=3
) -> np.ndarray:
    """Render an image along with its annotations."""
    assert annotated.image
    image = annotated.to_numpy()
    width, height = annotated.width, annotated.height

    for annotation in annotated.annotations:
        bbox = annotation.bbox.denormalize(width, height).to_int()
        cv2.rectangle(image, bbox.top_left, bbox.bottom_right, color, line_width)
    return image


def render_annotations(
    image: np.ndarray,
    annotations: List[Annotation]
):
    """Renders bounding boxes onto an image, optionally with labels and probabilities."""
    for annotation in annotations:
        box = annotation.bbox.to_int()
        cv2.rectangle(image, box.top_left, box.bottom_right, (255, 255, 0), 4)

        label = annotation.class_name
        probability = annotation.confidence

        if label or probability:
            text = f"{label} {probability:.2f}".strip()
            cv2.putText(
                image,
                text,
                (box.top_left[0] + 20, box.top_left[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # font scale
                (255, 0, 255),
                2,
            )  # line type
