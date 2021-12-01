from random import Random

import numpy as np

from ...dataloading import DataLoader, LoaderWrapper
from ..dataloading import ImageAugmentationLoader
from ..image import ImageSize, get_image_size
from .patching import get_patch, get_patches_per_image


class SegmentationAugmentationLoader(ImageAugmentationLoader):
    def __init__(self, loader: DataLoader, augmentation_args, seed=None):
        """
        Loader that wraps another loader and augments both inputs and labels using
        keras.ImageDataGenerator.

        Intended for image-to-image segmentation tasks.

        :param augmentation_args: constructor arguments for keras.ImageDataGenerator
        """
        super().__init__(
            loader, augmentation_args=augmentation_args, seed=seed, augment_label=True
        )


class PatchSampler(LoaderWrapper):
    """
    Loader that wraps a segmentation loader and returns randomly cropped patches from it.
    Expects that the input loader returns tuples (image, label).
    """

    def __init__(self, loader: DataLoader, size: int, stride: int):
        super().__init__(loader)
        self.image_size = get_image_size_from_loader(loader)
        self.size = size
        self.stride = stride
        self.patches_per_image = get_patches_per_image(
            self.image_size, (self.size, self.size), (self.stride, self.stride)
        )
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


class FilteredPatchSampler(PatchSampler):
    """
    Loader that returns patches of input images and masks.
    Filters patches where masks are black with a certain probability.
    """

    def __init__(
        self, loader: DataLoader, size: int, stride: int, keep_black_probability: float
    ):
        super().__init__(loader, size, stride)
        self.keep_black_probability = keep_black_probability
        self.patch_indices = np.arange(0, self.patches_per_image)

    def __getitem__(self, index):
        x, y = self.loader[index]
        np.random.shuffle(self.patch_indices)
        for patch_index in self.patch_indices:
            y_patch = self.get_patch(y, patch_index)
            all_empty = np.all(y_patch == 0)
            if all_empty and self.random.random() > self.keep_black_probability:
                continue
            x_patch = self.get_patch(x, patch_index)
            return x_patch, y_patch

        return super().__getitem__(index)
