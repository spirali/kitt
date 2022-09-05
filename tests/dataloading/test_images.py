import numpy as np

from kitt.dataloading import DataLoader
from kitt.image.dataloading import ImageAugmentationLoader, ImageLoader, iterate_images
from kitt.image.image.tf import load_image

from ..conftest import data_path


def test_iterate_images_directory():
    assert set(iterate_images(data_path("dataset"))) == {
        str(data_path("dataset/1.jpeg")),
        str(data_path("dataset/2.jpeg")),
        str(data_path("dataset/3.jpeg")),
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


def test_image_loader_no_paths():
    images = [
        data_path("dataset/1.jpeg"),
        data_path("dataset/2.jpeg"),
        data_path("dataset/3.jpeg"),
    ]

    loader = ImageLoader(images, with_path=False)
    assert len(loader) == len(images)

    for (image_path, img) in zip(images, loader):
        assert np.all(load_image(image_path) == img)


def test_augmentation_loader():
    class Loader(DataLoader):
        def __len__(self):
            return 2

        def __getitem__(self, item):
            image = np.random.randn(3, 3, 3)
            return image, np.array([1, 2, 3])

    loader = Loader()
    loader = ImageAugmentationLoader(
        loader,
        dict(
            rotation_range=10.0,
            width_shift_range=0.02,
            height_shift_range=0.02,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode="constant",
        ),
    )
    for (x, y) in loader:
        assert np.allclose(y, [1, 2, 3])
