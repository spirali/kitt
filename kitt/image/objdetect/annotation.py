from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Union

import numpy as np
from PIL.Image import Image


@dataclass(frozen=False)
class BBoxBase:
    """
    Represents a base class for bounding boxes.
    Use either `BBox` or `NormalizedBBox`.

    The origin of the coordinate format is in the "top-left corner", i.e.:
    X grows to the right
    Y grows to the bottom
    """

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    __slots__ = ["xmin", "xmax", "ymin", "ymax"]

    def __post_init__(self):
        if type(self) is BBoxBase:
            raise Exception("Please create either BBox or NormalizedBBox")
        assert self.xmax > self.xmin
        assert self.ymax > self.ymin

    @classmethod
    def from_x1y1x2y2(cls, x1: float, y1: float, x2: float, y2: float):
        return cls(xmin=x1, xmax=x2, ymin=y1, ymax=y2)

    @classmethod
    def from_xywh(cls, x: float, y: float, width: float, height: float):
        return cls(xmin=x, xmax=x + width, ymin=y, ymax=y + height)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    def x1y1x2y2(self) -> Tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def xywh(self) -> Tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.width, self.height)

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def top_left(self) -> Tuple[float, float]:
        return (self.xmin, self.ymin)

    @property
    def bottom_right(self) -> Tuple[float, float]:
        return (self.xmax, self.ymax)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.xmin + (self.width / 2), self.ymin + (self.height / 2))

    @property
    def area(self) -> float:
        return self.width * self.height

    def move(self, x: float, y: float):
        return BBox(self.xmin + x, self.xmax + x, self.ymin + y, self.ymax + y)

    def __repr__(self):
        return repr(self.as_tuple())


class BBox(BBoxBase):
    def normalize(self, width: float, height: float) -> "NormalizedBBox":
        return NormalizedBBox(
            self.xmin / width, self.xmax / width, self.ymin / height, self.ymax / height
        )

    def to_int(self) -> "BBox":
        return BBox(*(int(v) for v in self.as_tuple()))


class NormalizedBBox(BBoxBase):
    """
    BoundingBox with normalized coordinates in the range [0, 1].
    """

    def __post_init__(self):
        super().__post_init__()
        assert 0 <= self.xmin <= 1
        assert 0 <= self.xmax <= 1
        assert 0 <= self.ymin <= 1
        assert 0 <= self.ymax <= 1

    def denormalize(self, width: float, height: float) -> "BBox":
        return BBox(
            self.xmin * width, self.xmax * width, self.ymin * height, self.ymax * height
        )


class AnnotationType(Enum):
    GROUND_TRUTH = 1
    PREDICTION = 2


# Annotations need to be comparable by identity, not value
@dataclass(frozen=False, eq=False)
class Annotation:
    class_name: str
    bbox: NormalizedBBox
    type: AnnotationType = AnnotationType.GROUND_TRUTH
    confidence: float = None

    @staticmethod
    def ground_truth(class_name: str, bbox: NormalizedBBox) -> "Annotation":
        return Annotation(
            class_name=class_name, bbox=bbox, type=AnnotationType.GROUND_TRUTH
        )

    @staticmethod
    def prediction(
        class_name: str, bbox: NormalizedBBox, confidence: float
    ) -> "Annotation":
        return Annotation(
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

    image: Union[Image, None]
    filename: str
    annotations: List[Annotation]

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.image)

    def __repr__(self):
        return "{}: {}".format(self.filename, self.annotations)
