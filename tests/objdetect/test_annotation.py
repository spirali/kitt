import numpy as np
import pytest

from kitt.image.image.tf import load_image
from kitt.image.objdetect.annotation import AnnotatedImage

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
