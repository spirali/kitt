import numpy as np

from kitt.image.image.tf import load_image
from kitt.image.objdetect.annotation import AnnotatedImage

from ..conftest import data_path


def test_load_img_shape():
    img = load_image(data_path("example.jpeg"))
    annotation = AnnotatedImage(img, "xyz", [])
    assert annotation.width == 500
    assert annotation.height == 375


def test_img_conversion():
    img = load_image(data_path("example.jpeg"))
    annotation = AnnotatedImage(img, "xyz", [])
    image_np = annotation.to_numpy()
    assert isinstance(image_np, np.ndarray)
    assert image_np.shape == (375, 500, 3)
    image_pil = annotation.to_pillow()
    assert image_pil.width == 500
    assert image_pil.height == 375
