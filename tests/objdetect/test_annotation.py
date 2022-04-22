import numpy as np
import pytest

from kitt.image.image.tf import load_image
from kitt.image.objdetect.annotation import AnnotatedBBox, AnnotatedImage
from kitt.image.objdetect.bbox import BBox

from ..conftest import data_path


def test_annotated_image_shape():
    annotation = AnnotatedImage(size=(500, 375))
    assert annotation.width == 500
    assert annotation.height == 375


def test_annotated_image_shape_from_image():
    img = load_image(data_path("example.jpeg"))
    annotation = AnnotatedImage.from_image(img)
    assert annotation.width == 500
    assert annotation.height == 375


def test_annotated_image_wrong_image_shape():
    img = load_image(data_path("example.jpeg"))
    with pytest.raises(Exception):
        AnnotatedImage(image=img, size=(100, 100))


def test_img_conversion():
    img = load_image(data_path("example.jpeg"))
    annotation = AnnotatedImage.from_image(img)
    image_np = annotation.to_numpy()
    assert isinstance(image_np, np.ndarray)
    assert image_np.shape == (375, 500, 3)
    image_pil = annotation.to_pillow()
    assert image_pil.width == 500
    assert image_pil.height == 375


def test_map_bbox():
    bbox = AnnotatedBBox("foo", BBox(1, 2, 3, 4))
    bbox2 = bbox.map_bbox(
        lambda b: BBox(b.xmin * 2, b.xmax * 2, b.ymin * 2, b.ymax * 2)
    )
    assert bbox.class_name == bbox2.class_name
    assert bbox2.bbox.x1y1x2y2() == (2, 6, 4, 8)
