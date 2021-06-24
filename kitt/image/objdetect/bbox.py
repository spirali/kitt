from dataclasses import dataclass
from typing import Tuple


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
        return self.from_x1y1x2y2(
            self.xmin + x, self.ymin + y, self.xmax + x, self.ymax + y
        )

    def clip(self, x_max, y_max, x_min=0.0, y_min=0.0):
        xmin = max(x_min, self.xmin)
        xmax = min(x_max, self.xmax)
        ymin = max(y_min, self.ymin)
        ymax = min(y_max, self.ymax)
        return self.from_x1y1x2y2(xmin, ymin, xmax, ymax)

    def rescale_at_center(self, scale: float):
        """
        Scales the bounding box so that it is `scale`-times lager or smaller.
        Its center will stay at the same position as before.
        """
        center = self.center
        width = self.width * scale
        height = self.height * scale
        x = center[0] - width / 2
        y = center[1] - height / 2
        x2 = x + width
        y2 = y + height
        args = [self._clip_dimension(v) for v in (x, y, x2, y2)]
        return self.from_x1y1x2y2(*args)

    def _clip_dimension(self, value: float) -> float:
        return value

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

    def _clip_dimension(self, value: float) -> float:
        return max(min(value, 1.0), 0.0)