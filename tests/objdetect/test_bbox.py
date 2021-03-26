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
