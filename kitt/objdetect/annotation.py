from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from PIL.Image import Image


@dataclass
class BoundingBox:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def data(self) -> Tuple[float, float, float, float]:
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

    def denormalize(self, width: float, height: float) -> "BoundingBox":
        return BoundingBox(
            self.xmin * width, self.xmax * width, self.ymin * height, self.ymax * height
        )

    def normalize(self, width: float, height: float) -> "BoundingBox":
        return BoundingBox(
            self.xmin / width, self.xmax / width, self.ymin / height, self.ymax / height
        )

    def to_int(self) -> "BoundingBox":
        return BoundingBox(*(int(v) for v in self.data))

    def __repr__(self):
        return repr(self.data)


@dataclass
class Annotation:
    class_name: str
    bbox: BoundingBox


@dataclass
class AnnotatedImage:
    """
    Annotated image with a list of normalized bounding boxes.
    """
    image: Union[Image, None]
    filename: str
    annotations: Tuple[Annotation]

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
