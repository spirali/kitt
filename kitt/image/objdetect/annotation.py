from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Union

import numpy as np
from PIL import Image

from ..image import ImageSize, get_image_size
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

    def map_bbox(self, map_fn: Callable[[BBoxBase], BBoxBase]) -> "AnnotatedBBox":
        bbox = map_fn(self.bbox)
        return AnnotatedBBox(
            class_name=self.class_name,
            bbox=bbox,
            type=self.type,
            confidence=self.confidence,
        )

    def __post_init__(self):
        if self.type == AnnotationType.PREDICTION:
            assert self.confidence is not None


@dataclass
class AnnotatedImage:
    """
    Annotated image with a list of bounding boxes.
    """

    size: ImageSize
    annotations: List[AnnotatedBBox] = field(default_factory=list)
    image: Optional[np.ndarray] = None
    filename: Optional[str] = None

    @staticmethod
    def from_image(
        image: np.ndarray, annotations: List[AnnotatedBBox] = None, filename: str = None
    ):
        size = get_image_size(image)
        return AnnotatedImage(
            image=image, size=size, annotations=annotations, filename=filename
        )

    def __post_init__(self):
        self.annotations = self.annotations or []
        if self.image is not None:
            assert get_image_size(self.image) == self.size

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    def to_numpy(self) -> np.ndarray:
        return self.image

    def to_pillow(self) -> Union[Image.Image, None]:
        if self.image is not None:
            assert self.image.dtype == np.uint8
            return Image.fromarray(self.image.astype("uint8"))
        return None

    def __repr__(self):
        return f"{self.size} (at {self.filename}): {self.annotations}"
