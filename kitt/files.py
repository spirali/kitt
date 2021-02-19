import glob
import itertools
import os
from typing import Iterable, Union


def iterate_files(directory: str, extension: str, prefix: str = "") -> Iterable[str]:
    """Recursively return all files with the given `prefix` and `extension`
    that belong inside the given `directory`."""
    extension = extension.lstrip(".")
    for xml in sorted(
        glob.glob(os.path.join(directory, f"**/{prefix}*.{extension}"), recursive=True)
    ):
        yield xml


def iterate_directories(
    directories: Union[str, Iterable[str]], extension: str, prefix: str = ""
) -> Iterable[str]:
    """Recursively return all files with the given `extension` from a list of directories."""
    if isinstance(directories, str):
        directories = (directories,)

    return itertools.chain.from_iterable(
        iterate_files(directory, extension, prefix) for directory in directories
    )


def ensure_directory(path: str):
    if os.path.isfile(path) and not os.path.isdir(path):
        path = os.path.dirname(path)
    os.makedirs(path, exist_ok=True)
