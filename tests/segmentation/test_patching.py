import numpy as np

from kitt.dataloading import DataLoader
from kitt.image.segmentation.dataloading import (
    PatchingSamplerLoader,
    SegmentationAugmentationLoader,
)
from kitt.image.segmentation.patching import get_patch, get_patches_per_dimension


def test_patches_per_dimension():
    assert get_patches_per_dimension(1024, size=256, stride=256) == 4
    assert get_patches_per_dimension(1024, size=256, stride=128) == 7


def test_get_patch():
    image = np.array(range(64)).reshape((8, 8))

    assert (
        get_patch(image, 4, 4, 0).flatten()
        == [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
    ).all()
    assert (
        get_patch(image, 4, 4, 1).flatten()
        == [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31]
    ).all()
    assert (
        get_patch(image, 4, 4, 2).flatten()
        == [32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59]
    ).all()
    assert (
        get_patch(image, 4, 4, 3).flatten()
        == [36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63]
    ).all()

    assert (
        get_patch(image, 4, 2, 1).flatten()
        == [2, 3, 4, 5, 10, 11, 12, 13, 18, 19, 20, 21, 26, 27, 28, 29]
    ).all()
    assert (
        get_patch(image, 4, 2, 3).flatten()
        == [16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43]
    ).all()
    assert (
        get_patch(image, 4, 2, 8).flatten()
        == [36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63]
    ).all()


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


def test_patching_loader():
    class Loader(DataLoader):
        def __init__(self, images, masks):
            assert len(images) == len(masks)
            self.images = images
            self.masks = masks

        def __len__(self):
            return len(self.images)

        def __getitem__(self, index):
            return self.images[index], self.masks[index]

    items = []
    dim = 4
    count = dim * dim
    for i in range(8):
        start = i * count
        item = np.array(list(range(start, start + count))).reshape((dim, dim))
        items.append(item)

    def contained_within(image, subimage):
        def check(a, b, upper_left):
            ul_row = upper_left[0]
            ul_col = upper_left[1]
            b_rows, b_cols = b.shape
            a_slice = a[ul_row : ul_row + b_rows, :][:, ul_col : ul_col + b_cols]
            if a_slice.shape != b.shape:
                return False
            return (a_slice == b).all()

        def find_slice(big_array, small_array):
            upper_left = np.argwhere(big_array == small_array[0, 0])
            for ul in upper_left:
                if check(big_array, small_array, ul):
                    return True
            return False

        return find_slice(image, subimage)

    images = items[:4]
    masks = items[4:]
    loader = Loader(images, masks)
    loader = PatchingSamplerLoader(loader, size=2, stride=2)
    for (index, (x, y)) in enumerate(loader):
        assert x.shape == (2, 2)
        assert y.shape == (2, 2)
        assert contained_within(images[index], x)
        assert contained_within(masks[index], y)
