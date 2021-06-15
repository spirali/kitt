import numpy as np
import pytest

from kitt.image.objdetect.annotation import BBox, BBoxBase, NormalizedBBox


def test_create_invalid_normalized_bbox():
    with pytest.raises(Exception):
        NormalizedBBox(xmin=5, xmax=10, ymin=1, ymax=2)


def test_create_bbox_base_directly():
    with pytest.raises(Exception):
        BBoxBase(xmin=0, xmax=1, ymin=0, ymax=1)


def test_normalize_bbox():
    bbox = BBox(xmin=0, xmax=50, ymin=50, ymax=75)
    bbox = bbox.normalize(100, 100)
    assert bbox.as_tuple() == (0, 0.5, 0.5, 0.75)


def test_denormalize_bbox():
    bbox = NormalizedBBox(xmin=0, xmax=1, ymin=0.5, ymax=0.75)
    bbox = bbox.denormalize(100, 100)
    assert bbox.as_tuple() == (0, 100, 50, 75)


def test_clip_bbox():
    bbox = BBox(xmin=-10.3, xmax=50.0, ymin=-20.3, ymax=75.9)
    bbox = bbox.clip(20, 30, 0, 0)
    assert bbox.as_tuple() == (0, 20, 0, 30)
    bbox = BBox(xmin=0, xmax=20, ymin=0, ymax=30)
    bbox = bbox.clip(20, 30, 0, 0)
    assert bbox.as_tuple() == (0, 20, 0, 30)
    bbox = BBox(xmin=0, xmax=20, ymin=0, ymax=30)
    bbox = bbox.clip(21, 31, 0, 0)
    assert bbox.as_tuple() == (0, 20, 0, 30)


def test_invalid_bbox():
    with pytest.raises(Exception):
        BBox(xmin=0, xmax=0, ymin=0.5, ymax=0.6)

    with pytest.raises(Exception):
        BBox(xmin=1, xmax=0, ymin=0.3, ymax=0.5)


def test_from_xywh():
    bbox = BBox.from_xywh(5, 6, 10, 30)
    assert bbox.xmin == 5
    assert bbox.xmax == 15
    assert bbox.ymin == 6
    assert bbox.ymax == 36


def test_to_xywh():
    bbox = BBox(xmin=5, xmax=6, ymin=10, ymax=30)
    x, y, w, h = bbox.xywh()
    assert x == 5
    assert w == 1
    assert y == 10
    assert h == 20


def test_class_constructor():
    bbox = BBox.from_xywh(1, 1, 5, 5)
    assert isinstance(bbox, BBox)

    bbox = NormalizedBBox.from_xywh(0.1, 0.1, 0.5, 0.5)
    assert isinstance(bbox, NormalizedBBox)


def test_rescale_at_center():
    bbox = BBox.from_xywh(100, 200, 50, 100)
    assert bbox.rescale_at_center(2).xywh() == (75, 150, 100, 200)


def test_rescale_at_center_normalized():
    bbox = NormalizedBBox.from_x1y1x2y2(0.1, 0.2, 0.5, 0.3)
    assert np.allclose(bbox.rescale_at_center(2).x1y1x2y2(), (0, 0.15, 0.7, 0.35))

    bbox = NormalizedBBox.from_x1y1x2y2(0.2, 0.1, 0.3, 0.5)
    assert np.allclose(bbox.rescale_at_center(2).x1y1x2y2(), (0.15, 0, 0.35, 0.7))
