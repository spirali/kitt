import cv2
import matplotlib.pyplot as plt
import numpy as np

from kitt.image.image import resize_if_needed, resize_to_height, resize_to_width
from kitt.image.image.tf import load_image
from kitt.image.plot import render_plt_to_cv
from kitt.image.segmentation.image import polygons_to_binary_mask

from .conftest import check_image_equality, data_path


def test_load_image_rgb():
    img = load_image(data_path("example.jpeg"))
    assert img.shape == (375, 500, 3)
    assert np.max(img) > 1.0


def test_load_image_bgr():
    img2 = load_image(data_path("example.jpeg"), color_mode="bgr")
    check_image_equality(
        cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), data_path("example.jpeg")
    )


def test_load_image_grayscale():
    img = load_image(data_path("example.jpeg"), color_mode="grayscale")
    assert img.shape == (375, 500)


def test_load_image_resize():
    img = load_image(data_path("example.jpeg"), target_size=(224, 224))
    assert img.shape == (224, 224, 3)


def test_load_image_resize_non_uniform():
    img = load_image(data_path("example.jpeg"), target_size=(256, 512))
    assert img.shape == (512, 256, 3)


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


def test_resize_keep_aspect_ratio_vertically_upsample():
    img = np.zeros((8, 4))
    color = 250
    result_img = np.zeros((10, 4))
    result_img[[0, -1]] = color
    img2 = resize_if_needed(img, (4, 10), True, color)
    assert np.array_equal(result_img, img2)


def test_resize_keep_aspect_ratio_horizontally_upsample():
    img = np.zeros((4, 8))
    color = 250
    result_img = np.zeros((4, 10))
    result_img[:, [0, -1]] = color
    img2 = resize_if_needed(img, (10, 4), True, color)
    assert np.array_equal(result_img, img2)


def test_resize_keep_aspect_ratio_vertically_downsample():
    img = np.zeros((8, 4))
    color = 250
    result_img = np.zeros((4, 4))
    result_img[:, [0, -1]] = color
    img2 = resize_if_needed(img, (4, 4), True, color)
    assert np.array_equal(result_img, img2)


def test_resize_keep_aspect_ratio_horizontally_downsample():
    img = np.zeros((4, 8))
    color = 250
    result_img = np.zeros((4, 4))
    result_img[[0, -1]] = color
    img2 = resize_if_needed(img, (4, 4), True, color)
    assert np.array_equal(result_img, img2)


def test_resize_keep_aspect_ratio_mixed_downsample():
    img = np.zeros((4, 8))
    color = 250
    result_img = np.zeros((8, 4))
    result_img[[0, 1, 2, -3, -2, -1]] = color
    img2 = resize_if_needed(img, (4, 8), True, color)
    assert np.array_equal(result_img, img2)


def test_resize_to_width():
    img = np.zeros((8, 4))
    assert resize_to_width(img, 1).shape == (2, 1)
    assert resize_to_width(img, 2).shape == (4, 2)
    assert resize_to_width(img, 4).shape == (8, 4)
    assert resize_to_width(img, 8).shape == (16, 8)


def test_resize_to_height():
    img = np.zeros((8, 4))
    assert resize_to_height(img, 1).shape == (1, 1)
    assert resize_to_height(img, 2).shape == (2, 1)
    assert resize_to_height(img, 4).shape == (4, 2)
    assert resize_to_height(img, 8).shape == (8, 4)


def test_plot_to_image():
    x = [1, 2, 3]
    y = [5, 2, 8]

    plt.cla()
    plt.plot(x, y)

    image = render_plt_to_cv()
    check_image_equality(image, data_path("image/plot1.png"))
