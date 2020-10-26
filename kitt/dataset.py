from collections.abc import Iterable
from typing import Any, Callable, Tuple, Union

import pandas as pd

from .files import iterate_directories


def create_dataset(
    directories: Union[str, Tuple[str]],
    extension: str,
    val_split: float = None,
    label_fn=None,
):
    dataset = load_dataset(directories, extension)
    if val_split:
        dataset = validation_split(dataset, val_split)
    if label_fn:
        dataset = assign_labels(dataset, label_fn)
    return dataset


def load_dataset(directories: Union[str, Tuple[str]], extension: str) -> pd.DataFrame:
    if not isinstance(directories, Iterable):
        directories = (directories,)

    items = tuple(iterate_directories(directories, extension))
    return pd.DataFrame({"path": items})


def assign_labels(dataset: pd.DataFrame, fn: Callable[[str], Any]) -> pd.DataFrame:
    dataset = dataset.copy()
    dataset["label"] = dataset["path"].apply(fn)
    return dataset


def validation_split(
    dataset: pd.DataFrame, split: float, random_state=None
) -> pd.DataFrame:
    dataset = dataset.copy()
    dataset["train"] = True
    val_subset = dataset.sample(frac=split, random_state=random_state)
    dataset.loc[val_subset.index, "train"] = False
    return dataset
