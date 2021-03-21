from random import Random

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from ...dataloading import DataLoader, LoaderWrapper
from ..image import ImageSize, get_image_size
from .patching import get_patch, get_patches_per_dimension


class SegmentationAugmentationLoader(LoaderWrapper):
    def __init__(self, loader: DataLoader, augmentation_args, seed=None):
        """
        Loader that wraps another loader and augments both inputs and labels using
        keras.ImageDataGenerator.

        Expects that the input loader returns tuples (input, label).
        Intended for image-to-image segmentation tasks.

        :param augmentation_args: constructor arguments for keras.ImageDataGenerator
        """
        super().__init__(loader)
        self.seed = seed or 0
        self.augmentator = ImageDataGenerator(**augmentation_args)

    def __getitem__(self, index):
        self.seed += 1
        x, y = self.loader[index]
        return self._augment_image(x), self._augment_image(y)

    def _augment_image(self, image):
        assert image.ndim == 3
        image = np.expand_dims(image, axis=0)
        flow = self.augmentator.flow(image, batch_size=1, seed=self.seed, shuffle=False)
        return next(flow)[0]


class PatchingSamplerLoader(LoaderWrapper):
    """
    Loader that wraps a segmentation loader and returns randomly cropped patches from it.
    Expects that the input loader returns tuples (image, label).
    """

    def __init__(self, loader: DataLoader, size: int, stride: int):
        super().__init__(loader)
        self.image_size = get_image_size_from_loader(loader)
        self.size = size
        self.stride = stride
        self.patches_per_image = get_patches_per_dimension(
            self.image_size[0], self.size, self.stride
        ) * get_patches_per_dimension(self.image_size[1], self.size, self.stride)
        self.random = Random()

    def get_patch(self, image: np.ndarray, index: int) -> np.ndarray:
        return get_patch(image, size=self.size, stride=self.stride, patch_index=index)

    def __getitem__(self, index):
        x, y = self.loader[index]
        patch_index = self.random.randint(0, self.patches_per_image - 1)
        x_patch = self.get_patch(x, patch_index)
        y_patch = self.get_patch(y, patch_index)
        return x_patch, y_patch


def get_image_size_from_loader(loader: DataLoader) -> ImageSize:
    item = loader[0]
    assert len(item) == 2
    image, mask = item
    image_size = get_image_size(image)
    assert image_size == get_image_size(mask)
    return image_size
