from collections import defaultdict

import numpy as np

from kitt.dataloading import DataLoader
from kitt.image.segmentation.dataloading import FilteredPatchSampler, PatchLoader
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


class Loader(DataLoader):
    def __init__(self, images, masks):
        assert len(images) == len(masks)
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.masks[index]


def test_patching_loader_size():
    images = np.zeros((4, 4, 3))
    labels = images
    loader = Loader([images], [labels])

    assert len(PatchLoader(loader, size=2, stride=2)) == 4
    assert len(PatchLoader(loader, size=2, stride=1)) == 9


def test_patching_loader():
    images = []
    labels = []
    expected_x_patches = defaultdict(int)

    """
    Generate images like this:
    1122
    3344
    """
    dim = 4
    for image_index in range(8):
        image = np.zeros((dim * dim, dim * dim))
        for patch_row in range(dim):
            for patch_col in range(dim):
                value = image_index + patch_row * dim + patch_col
                patch = np.full((dim, dim), value)
                expected_x_patches[tuple(patch.ravel())] += 1
                image[
                    patch_row * dim : (patch_row + 1) * dim,
                    patch_col * dim : (patch_col + 1) * dim,
                ] = patch
        images.append(image)
        labels.append(image * 2)

    loader = Loader(images, labels)
    loader = PatchLoader(loader, size=dim, stride=dim)
    for (index, (x, y)) in enumerate(loader):
        assert x.shape == (dim, dim)
        assert y.shape == (dim, dim)
        assert (y == x * 2).all()

        key = tuple(x.ravel())
        assert key in expected_x_patches
        expected_x_patches[key] -= 1
    assert all(v == 0 for v in expected_x_patches.values())


def test_filtered_patching_loader():
    image = np.zeros((256, 256), dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[:128, :] = 255

    loader = Loader([image], [mask])
    filtered = FilteredPatchSampler(
        loader, size=64, stride=64, keep_black_probability=0
    )

    for i in range(5):
        for (x, y) in filtered:
            assert not np.all(y == 0)

    filtered = FilteredPatchSampler(
        loader, size=64, stride=64, keep_black_probability=1
    )
    black_found = False
    for i in range(10):
        for (x, y) in filtered:
            if np.all(y == 0):
                black_found = True
                break
    assert black_found


def test_filtered_patching_loader_all_black():
    image = np.zeros((256, 256), dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)

    loader = Loader([image], [mask])
    filtered = FilteredPatchSampler(
        loader, size=64, stride=64, keep_black_probability=0
    )
    for (x, y) in filtered:
        assert np.all(y == 0)
