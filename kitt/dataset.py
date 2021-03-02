import os
from collections.abc import Iterable
from typing import Any, Callable, Tuple, Union

import pandas as pd

from .files import iterate_directories


def create_dataset(
    directories: Union[str, Tuple[str]],
    extension: str,
    sample_prefix: str = "",
    val_split: float = None,
    label_fn=None,
):
    dataset = load_dataset(directories, extension, sample_prefix)
    if val_split:
        dataset = assign_validation_split(dataset, val_split)
    if label_fn:
        dataset = assign_labels(dataset, label_fn)
    return dataset


def load_dataset(
    directories: Union[str, Tuple[str]], extension: str, sample_prefix: str = ""
) -> pd.DataFrame:
    if not isinstance(directories, Iterable):
        directories = (directories,)

    items = (
        os.path.abspath(path)
        for path in tuple(iterate_directories(directories, extension, sample_prefix))
    )
    return pd.DataFrame({"path": items})


def assign_labels(dataset: pd.DataFrame, fn: Callable[[str], Any]) -> pd.DataFrame:
    dataset = dataset.copy()
    dataset["label"] = dataset["path"].apply(fn)
    return dataset


def assign_validation_split(
    dataset: pd.DataFrame, split: float, random_state=None
) -> pd.DataFrame:
    dataset = dataset.copy()
    dataset["train"] = True
    val_subset = dataset.sample(frac=split, random_state=random_state)
    dataset.loc[val_subset.index, "train"] = False
    return dataset


def split_dataset(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into training and validation datasets, based on the "train" column"""
    training = dataset[dataset["train"]]
    validation = dataset[~dataset["train"]]
    return (training, validation)
