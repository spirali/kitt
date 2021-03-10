import cv2
import numpy as np

from kitt.image.image import load_image
from kitt.image.segmentation.image import polygons_to_binary_mask
from tests.conftest import data_path


def test_load_image_rgb():
    img = load_image(data_path("example.jpeg"))
    assert img.shape == (375, 500, 3)
    assert np.max(img) > 1.0


def test_load_image_bgr():
    img = load_image(data_path("example.jpeg"))
    img2 = load_image(data_path("example.jpeg"), color_mode="bgr")
    assert (cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) == img).all()


def test_load_image_grayscale():
    img = load_image(data_path("example.jpeg"), color_mode="grayscale")
    assert img.shape == (375, 500)


def test_load_image_resize():
    img = load_image(data_path("example.jpeg"), target_size=(224, 224))
    assert img.shape == (224, 224, 3)


def test_load_image_normalize():
    img = load_image(data_path("example.jpeg"), normalize=True)
    assert np.max(img) <= 1.0


def test_polygons_to_mask():
    mask = polygons_to_binary_mask((16, 16), [[(2, 2), (2, 4), (4, 4), (4, 2)]])
    image = np.zeros((16, 16), dtype=np.float)
    image[2:5, 2:5] = 1
    assert (image == mask).all()
