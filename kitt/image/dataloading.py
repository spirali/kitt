import os
from typing import List

from ..dataloading import ListDataLoader
from ..files import iterate_files
from .image.tf import load_image


class ImageLoader(ListDataLoader):
    """
    Loads images using the `load_image` function.

    Returns pairs (image_path, image_data).
    """

    def __init__(self, paths: List[str], **load_image_args):
        super().__init__(paths)
        self.load_image_args = load_image_args

    def __getitem__(self, index):
        path = super().__getitem__(index)
        return (path, load_image(path, **self.load_image_args))


def iterate_images(path: str):
    """Return an iterator that finds all JPG/PNG images under recursively the given path."""

    def filter_fn(p):
        return os.path.splitext(p)[1] in (".jpg", ".jpeg", ".png")

    if os.path.isfile(path) and filter_fn(path):
        yield path
    elif os.path.isdir(path):
        yield from iterate_files(path, "jpg")
        yield from iterate_files(path, "jpeg")
        yield from iterate_files(path, "png")
