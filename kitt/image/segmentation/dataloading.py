from tensorflow.keras.preprocessing.image import ImageDataGenerator

from ...dataloading import SequenceWrapper


class SegmentationAugmentingSequence(SequenceWrapper):
    def __init__(self, sequence, augmentation_args, seed=None):
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
