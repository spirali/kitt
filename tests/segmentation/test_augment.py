import numpy as np

from kitt.dataloading import DataLoader
from kitt.image.segmentation.dataloading import SegmentationAugmentationLoader


def test_segmentation_loader_apply_same_augment():
    class Loader(DataLoader):
        def __len__(self):
            return 2

        def __getitem__(self, item):
            image = np.random.randn(3, 3, 3)
            return image, image

    loader = Loader()
    loader = SegmentationAugmentationLoader(
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
        assert np.allclose(x, y)
