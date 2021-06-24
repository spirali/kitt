from dataclasses import dataclass
from enum import Enum
from typing import List, Union

import numpy as np
from PIL import Image

from .bbox import BBoxBase


class AnnotationType(Enum):
    GROUND_TRUTH = 1
    PREDICTION = 2


# Annotations need to be comparable by identity, not value
@dataclass(frozen=True, eq=False)
class AnnotatedBBox:
    class_name: str
    bbox: BBoxBase
    type: AnnotationType = AnnotationType.GROUND_TRUTH
    confidence: float = None

    @staticmethod
    def ground_truth(class_name: str, bbox: BBoxBase) -> "AnnotatedBBox":
        return AnnotatedBBox(
            class_name=class_name, bbox=bbox, type=AnnotationType.GROUND_TRUTH
        )

    @staticmethod
    def prediction(
        class_name: str, bbox: BBoxBase, confidence: float
    ) -> "AnnotatedBBox":
        return AnnotatedBBox(
            class_name=class_name,
            bbox=bbox,
            type=AnnotationType.PREDICTION,
            confidence=confidence,
        )

    def __post_init__(self):
        if self.type == AnnotationType.PREDICTION:
            assert self.confidence is not None


@dataclass
class AnnotatedImage:
    """
    Annotated image with a list of normalized bounding boxes.
    """

    image: Union[np.ndarray, None]
    filename: str
    annotations: List[AnnotatedBBox]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]

    def to_numpy(self) -> np.ndarray:
        return self.image

    def to_pillow(self) -> Union[Image.Image, None]:
        if self.image is not None:
            return Image.fromarray(self.image.astype("uint8"))
        return None

    def __repr__(self):
        return "{}: {}".format(self.filename, self.annotations)
