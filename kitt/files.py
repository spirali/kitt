import glob
import itertools
import os
from pathlib import Path
from typing import Callable, Iterable, Union

GenericPath = Union[Path, str]


def iterate_files(
    directory: GenericPath, extension: str, prefix: str = ""
) -> Iterable[str]:
    """Recursively return all files with the given `prefix` and `extension`
    that belong inside the given `directory`."""
    extension = extension.lstrip(".")
    for xml in sorted(
        glob.glob(os.path.join(directory, f"**/{prefix}*.{extension}"), recursive=True)
    ):
        yield xml


def iterate_directories(
    directories: Union[GenericPath, Iterable[GenericPath]],
    extension: str,
    prefix: str = "",
) -> Iterable[str]:
    """Recursively return all files with the given `extension` from a list of directories."""
    if isinstance(directories, str):
        directories = (directories,)

    return itertools.chain.from_iterable(
        iterate_files(directory, extension, prefix) for directory in directories
    )


def iterate_files_from(
    path: GenericPath, filter_fn: Callable[[Path], bool] = None
) -> Iterable[Path]:
    """
    Recursively return all files located at the given `path`.

    :param path: Path to a file or a directory.
    :param filter_fn Function that can be used to filter returned paths (e.g. by extension or
    prefix).
    """
    path = Path(path)
    if path.is_file():
        files = [path]
    elif path.is_dir() or path.is_symlink():
        files = glob.glob(f"{path}/**/*", recursive=True)
    else:
        raise Exception(f"Invalid path {path}")

    for path in sorted(files):
        path = Path(path).resolve()
        if not path.is_file():
            continue
        if filter_fn is not None and not filter_fn(path):
            continue
        yield path


def ensure_directory(path: GenericPath) -> Path:
    if os.path.isfile(path) and not os.path.isdir(path):
        path = os.path.dirname(path)
    os.makedirs(path, exist_ok=True)
    return Path(path).absolute()
