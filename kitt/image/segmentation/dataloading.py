from random import Random

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence

from ...dataloading import SequenceWrapper
from ..image import get_image_size
from .patching import get_patch, get_patches_per_dimension


class SegmentationAugmentingSequence(SequenceWrapper):
    def __init__(self, sequence: Sequence, augmentation_args, seed=None):
        """
        Sequence that wraps another sequence and augments both inputs and labels using
        keras.ImageDataGenerator.

        Intended for image-to-image segmentation tasks.

        All images within a batch are augmented with the same transform.

        :param augmentation_args: constructor arguments for keras.ImageDataGenerator
        """
        super().__init__(sequence)
        self.seed = seed or 0
        self.augmentator = ImageDataGenerator(**augmentation_args)

    def __getitem__(self, index):
        self.seed += 1
        return self._augment_batch(self.sequence[index])

    def _augment_images(self, images):
        flow = self.augmentator.flow(
            images, batch_size=images.shape[0], seed=self.seed, shuffle=False
        )
        return next(flow)

    def _augment_batch(self, batch):
        xs, ys = batch
        xs_augmented, ys_augmented = self._augment_images(xs), self._augment_images(ys)

        return (xs_augmented, ys_augmented)


class PatchingSequence(SequenceWrapper):
    """
    Sequence that wraps a segmentation sequence and generates randomly cropped patches from it.
    """

    def __init__(self, sequence: Sequence, size: int, stride: int):
        super().__init__(sequence)
        batch = sequence[0]
        image, mask = batch[0][0], batch[1][0]
        assert get_image_size(image) == get_image_size(mask)

        self.image_size = get_image_size(image)
        self.size = size
        self.stride = stride
        self.patches_per_image = get_patches_per_dimension(
            self.image_size[0], self.size, self.stride
        ) * get_patches_per_dimension(self.image_size[1], self.size, self.stride)
        self.random = Random()

    def __getitem__(self, index):
        xs, ys = self.sequence[index]
        batch_x = []
        batch_y = []
        for index in range(len(xs)):
            patch_index = self.random.randint(0, self.patches_per_image - 1)
            batch_x.append(
                get_patch(
                    xs[index],
                    size=self.size,
                    stride=self.stride,
                    patch_index=patch_index,
                )
            )
            batch_y.append(
                get_patch(
                    ys[index],
                    size=self.size,
                    stride=self.stride,
                    patch_index=patch_index,
                )
            )
        return np.array(batch_x), np.array(batch_y)
