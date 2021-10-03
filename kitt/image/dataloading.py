import os
from typing import List

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from ..dataloading import DataLoader, ListDataLoader, LoaderWrapper
from ..files import iterate_files
from .image.tf import load_image


class ImageLoader(ListDataLoader):
    """
    Loads images using the `load_image` function.

    Returns pairs (image_path, image_data).
    """

    def __init__(self, paths: List[str], with_path=True, **load_image_args):
        super().__init__(paths)
        self.load_image_args = load_image_args
        self.with_path = with_path

    def __getitem__(self, index):
        path = super().__getitem__(index)
        image = load_image(path, **self.load_image_args)
        if self.with_path:
            return (path, image)
        return image


class ImageAugmentationLoader(LoaderWrapper):
    def __init__(
        self, loader: DataLoader, augmentation_args, augment_label=False, seed=None
    ):
        """
        Loader that wraps another loader and augments images with keras.ImageDataGenerator.

        Expects that the input loader returns tuples (input, label).
        If `augment_label` is true, the labels will also be augmented.

        :param augmentation_args: constructor arguments for keras.ImageDataGenerator
        """
        super().__init__(loader)
        self.seed = seed or 0
        self.augmentator = ImageDataGenerator(**augmentation_args)
        self.augment_label = augment_label

    def __getitem__(self, index):
        self.seed += 1
        x, y = self.loader[index]

        x = self._augment_image(x)
        if self.augment_label:
            y = self._augment_image(y)

        return x, y

    def _augment_image(self, image):
        assert image.ndim == 3
        image = np.expand_dims(image, axis=0)
        flow = self.augmentator.flow(image, batch_size=1, seed=self.seed, shuffle=False)
        return next(flow)[0]


def iterate_images(path: str):
    """Return an iterator that finds all JPG/PNG images under recursively the given path."""

    def filter_fn(p):
        return os.path.splitext(p)[1] in (".jpg", ".jpeg", ".png")

    if os.path.isfile(path) and filter_fn(path):
        yield path
    elif os.path.isdir(path):
        yield from iterate_files(path, "jpg")
        yield from iterate_files(path, "jpeg")
        yield from iterate_files(path, "png")
