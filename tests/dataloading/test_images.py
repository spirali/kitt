import numpy as np

from kitt.image.dataloading import ImageLoader, iterate_images
from kitt.image.image.tf import load_image

from ..conftest import data_path


def test_iterate_images_directory():
    assert set(iterate_images(data_path("dataset"))) == {
        data_path("dataset/1.jpeg"),
        data_path("dataset/2.jpeg"),
        data_path("dataset/3.jpeg"),
    }


def test_iterate_images_file():
    assert set(iterate_images(data_path("dataset/1.jpeg"))) == {
        data_path("dataset/1.jpeg")
    }


def test_image_loader():
    images = [
        data_path("dataset/1.jpeg"),
        data_path("dataset/2.jpeg"),
        data_path("dataset/3.jpeg"),
    ]

    loader = ImageLoader(images)
    assert len(loader) == len(images)

    for (image_path, (path, img)) in zip(images, loader):
        assert image_path == path
        assert np.all(load_image(path) == img)
