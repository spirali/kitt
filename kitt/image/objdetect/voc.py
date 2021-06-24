import logging
import multiprocessing
import os
from collections import defaultdict
from typing import Iterable

import pandas as pd
from pascal_voc_tools import XmlParser

from ...files import ensure_directory, iterate_directories
from ..image import get_image_size
from ..image.tf import load_image
from .annotation import AnnotatedBBox, AnnotatedImage
from .bbox import BBox


def load_voc_from_directories(
    directories: Iterable[str], num_workers: int = None
) -> pd.DataFrame:
    """Load a pandas dataframe from directories containing VOC files"""
    items = defaultdict(list)
    num_workers = num_workers or multiprocessing.cpu_count()

    files = tuple(iterate_directories(directories, "xml"))
    with multiprocessing.Pool(num_workers) as pool:
        for result in pool.imap(_parse_dataset_item, files):
            if result is not None:
                file, image_file = result
                items["annotation"].append(file)
                items["image"].append(image_file)
    return pd.DataFrame(items)


def voc_to_annotated_image(path: str, load_image_flag=True) -> AnnotatedImage:
    parser = XmlParser()
    content = parser.load(path)

    def parse_annotation(elem, width, height) -> AnnotatedBBox:
        name = elem["name"]
        bbox = elem["bndbox"]
        bbox = tuple(int(bbox[key]) for key in ("xmin", "xmax", "ymin", "ymax"))

        return AnnotatedBBox(class_name=name, bbox=BBox(*bbox).normalize(width, height))

    def find_image(directory, annotation_dir, filename):
        current_dir = annotation_dir

        while True:
            parent = os.path.dirname(current_dir)
            if parent == "/" or parent == current_dir:
                break
            candidate = os.path.join(parent, directory, filename)
            if os.path.isfile(candidate):
                return candidate
            current_dir = parent

        fallback_path = os.path.join(annotation_dir, filename)
        if os.path.isfile(fallback_path):
            return fallback_path
        return None

    filename = content["filename"]
    size = content["size"]
    width, height = (int(size[k]) for k in ("width", "height"))
    annotations = list(
        parse_annotation(obj, width, height) for obj in content["object"]
    )

    img_filename = find_image(
        content["folder"], os.path.dirname(os.path.abspath(path)), filename
    )
    if not img_filename or not os.path.isfile(img_filename):
        raise FileNotFoundError("Couldn't find image {}".format(filename))
    image = None

    if load_image_flag:
        try:
            image = load_image(img_filename)
            assert get_image_size(image) == (width, height)
        except FileNotFoundError:
            raise FileNotFoundError("Couldn't load image {}".format(filename))
    return AnnotatedImage(
        annotations=annotations,
        size=(width, height),
        image=image,
        filename=img_filename,
    )


def annotated_image_to_voc(path: str, image: AnnotatedImage):
    ensure_directory(path)

    parser = XmlParser()
    parser.set_head(image.filename, image.width, image.height)
    for annotation in image.annotations:
        bbox = annotation.bbox.denormalize(image.width, image.height).to_int()
        parser.add_object(
            annotation.class_name, bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
        )
    parser.save(path)


def _parse_dataset_item(path):
    file = os.path.abspath(path)
    try:
        example = voc_to_annotated_image(file, load_image_flag=False)
        if len(example.annotations) == 0:
            logging.warning(f"Skipping {file} because it has no annotations")
            return None
        image_file = os.path.abspath(example.filename)
        return (file, image_file)
    except FileNotFoundError:
        logging.warning(f"Image for {file} not found")
