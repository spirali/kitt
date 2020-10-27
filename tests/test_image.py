import cv2
import numpy as np

from kitt.image.image import create_image_grid, load_image
from tests.conftest import data_path


def test_load_image_rgb():
    img = load_image(data_path("example.jpeg"))
    assert img.shape == (375, 500, 3)


def test_load_image_grayscale():
    img = load_image(data_path("example.jpeg"), color_mode="grayscale")
    assert img.shape == (375, 500)


def test_load_image_resize():
    img = load_image(data_path("example.jpeg"), target_size=(224, 224))
    assert img.shape == (224, 224, 3)


def test_grid():
    images = tuple(load_image(data_path(f"grid/{i}.jpg")) for i in range(1, 7))
    images = [
        cv2.resize(image, (image.shape[0] // 4, image.shape[1] // 4))
        for image in images
    ]
    grid = create_image_grid(images, cols=2, border=True)
    grid_reference = load_image(data_path("grid/grid.png"))

    assert grid_reference.shape == grid.shape
    assert not np.any(cv2.subtract(grid, grid_reference))
