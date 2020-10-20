from kitt.image import load_image_rgb
from tests.conftest import data_path


def test_load_image():
    img = load_image_rgb(data_path("example.jpeg"))
    assert img.shape == (375, 500, 3)


def test_load_image_resize():
    img = load_image_rgb(data_path("example.jpeg"), target_size=(224, 224))
    assert img.shape == (224, 224, 3)
