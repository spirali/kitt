import os

import pandas as pd
from conftest import data_path

from kitt.dataset import assign_validation_split, load_dataset, split_dataset


def test_load_dataset():
    dataset_path = os.path.abspath(data_path("dataset"))
    df = load_dataset(dataset_path, ".jpeg")
    assert sorted(df["path"]) == [
        os.path.join(dataset_path, f"{p}.jpeg") for p in ("1", "2", "3")
    ]


def test_val_split():
    dataset = pd.DataFrame({"x": [0] * 100})
    df = assign_validation_split(dataset, 0.2, random_state=0)
    assert len(df[~df["train"]]) == 20


def test_split_dataset():
    dataset = pd.DataFrame(
        {"x": [0, 1, 2, 3, 4], "train": [True, True, True, False, False]}
    )
    train, validation = split_dataset(dataset)
    assert list(train["x"]) == [0, 1, 2]
    assert list(validation["x"]) == [3, 4]
