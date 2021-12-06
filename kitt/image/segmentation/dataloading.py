import random
from typing import Tuple

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


class PatchLoader(LoaderWrapper):
    """
    Loader that wraps a segmentation loader and returns all patches from it.
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

    def __len__(self):
        return len(self.loader) * self.patches_per_image

    def get_patch(self, image: np.ndarray, index: int) -> np.ndarray:
        return get_patch(image, size=self.size, stride=self.stride, patch_index=index)

    def __getitem__(self, index: int):
        image_index, patch_index = self.get_indices(index)
        x, y = self.loader[image_index]
        x_patch = self.get_patch(x, patch_index)
        y_patch = self.get_patch(y, patch_index)
        return x_patch, y_patch

    def get_indices(self, index: int) -> Tuple[int, int]:
        """
        Returns the index of an input image and its corresponding patch index
        from the input `index`.
        """
        image_index = index // self.patches_per_image
        patch_index = index % self.patches_per_image
        return image_index, patch_index

    def split(self, test_ratio: float) -> Tuple["PatchLoader", "PatchLoader"]:
        a, b = self.loader.split(test_ratio)
        return (
            PatchLoader(a, size=self.size, stride=self.stride),
            PatchLoader(b, size=self.size, stride=self.stride),
        )


def get_image_size_from_loader(loader: DataLoader) -> ImageSize:
    item = loader[0]
    assert len(item) == 2
    image, mask = item
    image_size = get_image_size(image)
    assert image_size == get_image_size(mask)
    return image_size


class FilteredPatchSampler(PatchLoader):
    """
    Loader samples patches from input images and masks.
    It also filters patches where masks are empty (black) with a certain probability.
    """

    def __init__(
        self, loader: DataLoader, size: int, stride: int, keep_black_probability: float
    ):
        super().__init__(loader, size, stride)
        self.keep_black_probability = keep_black_probability
        self.max_filtered_patches = 16
        self.random = random.Random()

    def __getitem__(self, index):
        # Try to find a non-empty mask patch
        for _ in range(self.max_filtered_patches):
            random_index = self.random.randrange(len(self))
            image_index, patch_index = self.get_indices(random_index)

            x, y = self.loader[image_index]

            y_patch = self.get_patch(y, patch_index)
            all_empty = np.all(y_patch == 0)
            if all_empty and self.random.random() > self.keep_black_probability:
                continue
            x_patch = self.get_patch(x, patch_index)
            return x_patch, y_patch

        # No non-empty patch found, return patch from this index
        return super().__getitem__(index)
