import numpy as np

from ..image import ImageSize, get_image_size


def get_patches_per_dimension(dimension: int, size: int, stride: int) -> int:
    """
    Returns the number of patches that can be created in the given dimension without going over
    the dimension bounds.
    """
    assert size % stride == 0

    overlapping = (size // stride) - 1 if stride != size else 0
    return (dimension // stride) - overlapping


def get_patches_per_image(
    image_size: ImageSize, patch_size: ImageSize, patch_stride: ImageSize
) -> int:
    return get_patches_per_dimension(
        image_size[0], patch_size[0], patch_stride[0]
    ) * get_patches_per_dimension(image_size[1], patch_size[1], patch_stride[1])


def get_patch(
    image: np.ndarray, size: int, stride: int, patch_index: int
) -> np.ndarray:
    """
    Returns a single patch from the input image.

    The patch index should be a number in the interval [0, patch_count).
    Patches are numbered in row-major ordering.
    Patch 0, ..., <# of patch columns> is on the first row, etc.
    """
    width, height = get_image_size(image)
    patch_cols = get_patches_per_dimension(width, size, stride)
    patch_rows = get_patches_per_dimension(height, size, stride)

    assert patch_index < patch_rows * patch_cols
    row = patch_index // patch_cols
    col = patch_index % patch_cols

    return image[row * stride : row * stride + size, col * stride : col * stride + size]
