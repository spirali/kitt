import pytest

from kitt.image.objdetect.annotation import BBox, BBoxBase, NormalizedBBox


def test_create_invalid_normalized_bbox():
    with pytest.raises(Exception):
        NormalizedBBox(xmin=5.0, xmax=10, ymin=1, ymax=1)


def test_create_bbox_base_directly():
    with pytest.raises(Exception):
        BBoxBase(xmin=0, xmax=0, ymin=0, ymax=0)


def test_normalize_bbox():
    bbox = BBox(xmin=0, xmax=0, ymin=50, ymax=50)
    bbox = bbox.normalize(100, 100)
    assert bbox.as_tuple() == (0, 0, 0.5, 0.5)


def test_denormalize_bbox():
    bbox = NormalizedBBox(xmin=0, xmax=0, ymin=0.5, ymax=0.5)
    bbox = bbox.denormalize(100, 100)
    assert bbox.as_tuple() == (0, 0, 50, 50)
