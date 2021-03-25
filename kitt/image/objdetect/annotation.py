from enum import Enum
from typing import Tuple, Union, List

import numpy as np
from PIL.Image import Image
from dataclasses import dataclass


@dataclass(frozen=True)
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

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.xmin, self.xmax, self.ymin, self.ymax)

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


@dataclass
class Annotation:
    class_name: str
    bbox: NormalizedBBox
    annotation_type: AnnotationType = AnnotationType.GROUND_TRUTH
    confidence: float = None


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
