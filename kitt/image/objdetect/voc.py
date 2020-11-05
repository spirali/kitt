import logging
import multiprocessing
import os
from collections import defaultdict
from typing import Iterable

import pandas as pd
from PIL import Image
from pascal_voc_tools import XmlParser

from .annotation import AnnotatedImage, Annotation, BoundingBox
from ...files import ensure_directory, iterate_directories


def load_voc_from_directories(
    directories: Iterable[str], num_workers: int = None
) -> pd.DataFrame:
    """Load a pandas dataframe from directories containing VOC files"""
    items = defaultdict(lambda: [])
    num_workers = num_workers or multiprocessing.cpu_count()

    files = tuple(iterate_directories(directories, "xml"))
    with multiprocessing.Pool(num_workers) as pool:
        for result in pool.imap(_parse_dataset_item, files):
            if result is not None:
                file, image_file = result
                items["annotation"].append(file)
                items["image"].append(image_file)
    return pd.DataFrame(items)


def voc_to_annotated_image(path: str, load_image=True) -> AnnotatedImage:
    parser = XmlParser()
    content = parser.load(path)

    def parse_annotation(elem, width, height):
        name = elem["name"]
        bbox = elem["bndbox"]
        bbox = tuple(int(bbox[key]) for key in ("xmin", "xmax", "ymin", "ymax"))

        return Annotation(
            class_name=name, bbox=BoundingBox(*bbox).normalize(width, height)
        )

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
    annotations = tuple(
        parse_annotation(obj, width, height) for obj in content["object"]
    )

    img_filename = find_image(
        content["folder"], os.path.dirname(os.path.abspath(path)), filename
    )
    if not img_filename or not os.path.isfile(img_filename):
        raise FileNotFoundError("Couldn't find image {}".format(filename))
    image = None

    if load_image:
        try:
            image = Image.open(img_filename)
        except FileNotFoundError:
            raise FileNotFoundError("Couldn't load image {}".format(filename))
    return AnnotatedImage(image, img_filename, annotations)


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
        example = voc_to_annotated_image(file, load_image=False)
        if len(example.annotations) == 0:
            logging.warning(f"Skipping {file} because it has no annotations")
            return None
        image_file = os.path.abspath(example.filename)
        return (file, image_file)
    except FileNotFoundError:
        logging.warning(f"Image for {file} not found")
