from collections import defaultdict
from typing import List

import numpy as np

from kitt.dataloading import DataLoader
from kitt.image.image import get_image_size
from kitt.image.segmentation.dataloading import FilteredPatchSampler, PatchLoader
from kitt.image.segmentation.patching import (
    get_patch,
    get_patches_per_dimension,
    get_patches_per_image,
)


def test_patches_per_dimension():
    assert get_patches_per_dimension(1024, size=256, stride=256) == 4
    assert get_patches_per_dimension(1024, size=256, stride=128) == 7


def check_patches(
    image: np.ndarray, size: int, stride: int, expected: List[np.ndarray]
):
    assert len(expected) == get_patches_per_image(
        get_image_size(image), (size, size), (stride, stride)
    )
    for (patch_index, expected_patch) in enumerate(expected):
        patch = get_patch(image, size=size, stride=stride, patch_index=patch_index)
        if not (patch == expected_patch).all():
            raise Exception(f"Patch {patch_index} does not match")


def test_get_patch_same_dims_same_stride():
    image = np.array(range(64)).reshape((8, 8))

    check_patches(
        image, 4, 4, [image[:4, :4], image[:4, 4:8], image[4:8, :4], image[4:8, 4:8]]
    )


def test_get_patch_same_dims_different_stride():
    image = np.array(range(64)).reshape((8, 8))

    check_patches(
        image,
        4,
        2,
        [
            image[:4, :4],
            image[:4, 2:6],
            image[:4, 4:8],
            image[2:6, :4],
            image[2:6, 2:6],
            image[2:6, 4:8],
            image[4:8, :4],
            image[4:8, 2:6],
            image[4:8, 4:8],
        ],
    )


def test_get_patch_different_dims_same_stride():
    image = np.array(range(32)).reshape((4, 8))

    check_patches(
        image,
        4,
        4,
        [
            image[:4, :4],
            image[:4, 4:8],
        ],
    )

    image = np.array(range(32)).reshape((8, 4))

    check_patches(
        image,
        4,
        4,
        [
            image[:4, :4],
            image[4:8, :4],
        ],
    )


def test_get_patch_different_dims_different_stride():
    image = np.array(range(32)).reshape((4, 8))

    check_patches(
        image,
        4,
        2,
        [
            image[:4, :4],
            image[:4, 2:6],
            image[:4, 4:8],
        ],
    )

    image = np.array(range(32)).reshape((8, 4))

    check_patches(
        image,
        4,
        2,
        [
            image[:4, :4],
            image[2:6, :4],
            image[4:8, :4],
        ],
    )


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
