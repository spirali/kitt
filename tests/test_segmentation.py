import numpy as np
from conftest import check_image_equality, data_path

from kitt.dataloading import BatchGenerator
from kitt.image.image import load_image
from kitt.image.segmentation.dataloading import (
    PatchingSequence,
    SegmentationAugmentingSequence,
)
from kitt.image.segmentation.mask import (
    binarize_mask,
    color_bitmap_masks,
    overlay_masks,
)
from kitt.image.segmentation.patching import get_patch, get_patches_per_dimension


def test_color_bitmap_masks():
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[10:30, 20:40] = 255

    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[35:50, 40:60] = 255

    mask3 = np.zeros((100, 100), dtype=np.uint8)
    mask3[35:50, 70:90] = 255

    mask4 = np.zeros((100, 100), dtype=np.uint8)
    mask4[70:95, 10:80] = 255

    palette = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    colored_masks = color_bitmap_masks((mask1, mask2, mask3, mask4), palette)
    for (index, mask) in enumerate(colored_masks):
        check_image_equality(
            mask, data_path(f"segmentation/color_masks/mask-{index}.png")
        )


def test_overlay_masks():
    masks = [
        load_image(data_path(f"segmentation/color_masks/mask-{i}.png"))
        for i in range(4)
    ]
    background = np.zeros((100, 100, 3), dtype=np.uint8)
    background[:, :] = (50, 50, 50)

    check_image_equality(
        overlay_masks(background, masks, alpha=1.0),
        data_path("segmentation/color_masks/overlay-alpha-1.0.png"),
        delta=1,
    )
    check_image_equality(
        overlay_masks(background, masks, alpha=0.5),
        data_path("segmentation/color_masks/overlay-alpha-0.5.png"),
        delta=1,
    )


def test_overlay_empty_mask():
    mask = np.zeros((100, 100, 3), dtype=np.uint8)
    background = np.zeros((100, 100, 3), dtype=np.uint8)
    background[:, :] = (50, 50, 50)

    overlaid = overlay_masks(background.copy(), [mask])
    assert (background == overlaid).all()


def test_binarize_mask():
    mask = np.array(
        [
            [0.4, 0.6, 1.0],
            [0, 0.2, 0.1],
            [0.8, 0.3, 0.5],
        ]
    )
    binarized = binarize_mask(mask, threshold=0.5)
    assert (
        binarized
        == np.array(
            [
                [0, 1, 1],
                [0, 0, 0],
                [1, 0, 0],
            ],
            dtype=np.float32,
        )
    ).all()


def test_binarize_mask_multichannel():
    mask = np.array(
        [
            [[0.4, 0.6], [0.6, 0.9], [1.0, 1.0]],
            [[0, 1.0], [0.2, 0.1], [0.4, 0.3]],
            [[1.0, 0.8], [0.0, 0.0], [0.7, 0.2]],
        ]
    )
    binarized = binarize_mask(mask, threshold=0.5)
    assert (
        binarized
        == np.array(
            [
                [[0, 1], [1, 1], [1, 1]],
                [[0, 1], [0, 0], [0, 0]],
                [[1, 1], [0, 0], [1, 0]],
            ],
            dtype=np.float32,
        )
    ).all()


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
    class ImageGenerator(BatchGenerator):
        def __init__(self, length, batch_size: int):
            super().__init__(length, batch_size)

        def load_sample(self, index):
            image = np.random.randn(3, 3, 3)
            return image, image

    generator = SegmentationAugmentingSequence(
        ImageGenerator(4, 2),
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
    for (x, y) in generator:
        assert np.allclose(x, y)


def test_patching_data_loader():
    class ImageGenerator(BatchGenerator):
        def __init__(self, images, masks, batch_size):
            super().__init__(len(images), batch_size, shuffle=False)
            self.images = images
            self.masks = masks

        def load_sample(self, index):
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
    generator = ImageGenerator(images, masks, batch_size=2)
    generator = PatchingSequence(generator, size=2, stride=2)
    for (index, (xs, ys)) in enumerate(generator):
        assert xs.shape == (2, 2, 2)
        assert ys.shape == (2, 2, 2)
        for i in range(2):
            original_index = index * 2 + i
            assert contained_within(images[original_index], xs[i])
            assert contained_within(masks[original_index], ys[i])
