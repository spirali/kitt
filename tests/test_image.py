import cv2
import numpy as np

from kitt.image.image import load_image, resize_if_needed
from kitt.image.segmentation.image import polygons_to_binary_mask
from kitt.image.segmentation.mask import color_bitmap_masks
from tests.conftest import check_image_equality, data_path


def test_load_image_rgb():
    img = load_image(data_path("example.jpeg"))
    assert img.shape == (375, 500, 3)
    assert np.max(img) > 1.0


def test_load_image_bgr():
    img2 = load_image(data_path("example.jpeg"), color_mode="bgr")
    check_image_equality(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), data_path("example.jpeg"))


def test_load_image_grayscale():
    img = load_image(data_path("example.jpeg"), color_mode="grayscale")
    assert img.shape == (375, 500)


def test_load_image_resize():
    img = load_image(data_path("example.jpeg"), target_size=(224, 224))
    assert img.shape == (224, 224, 3)


def test_load_image_normalize():
    img = load_image(data_path("example.jpeg"), normalize=True)
    assert np.max(img) <= 1.0


def test_polygons_to_mask():
    mask = polygons_to_binary_mask((16, 16), [[(2, 2), (2, 4), (4, 4), (4, 2)]])
    image = np.zeros((16, 16), dtype=np.float)
    image[2:5, 2:5] = 1
    assert (image == mask).all()


def test_resize_avoid_useless_resize():
    img = np.zeros((4, 4))
    img2 = resize_if_needed(img, img.shape)
    assert img is img2


def test_resize_handle_dimensions_correctly():
    img = np.zeros((8, 4))
    img2 = resize_if_needed(img, (8, 4))
    assert img2.shape == (4, 8)


def test_resize_keep_last_dim():
    img = np.zeros((8, 4))
    img = img.reshape((8, 4, 1))
    img2 = resize_if_needed(img, (8, 4))
    assert img2.shape == (4, 8, 1)


def test_color_bitmap_masks():
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[10:30, 20:40] = 255

    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[35:50, 40:60] = 255

    mask3 = np.zeros((100, 100), dtype=np.uint8)
    mask3[35:50, 70:90] = 255

    mask4 = np.zeros((100, 100), dtype=np.uint8)
    mask4[70:95, 10:80] = 255

    palette = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0)
    ]
    colored_masks = color_bitmap_masks((mask1, mask2, mask3, mask4), palette)
    for (index, mask) in enumerate(colored_masks):
        check_image_equality(mask, data_path(f"segmentation/color_masks/mask-{index}.png"))
