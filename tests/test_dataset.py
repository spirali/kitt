import os

import pandas as pd

from conftest import data_path
from kitt.dataset import load_dataset, validation_split


def test_load_dataset():
    dataset_path = os.path.abspath(data_path("dataset"))
    df = load_dataset(dataset_path, ".jpeg")
    assert sorted(df["path"]) == [
        os.path.join(dataset_path, f"{p}.jpeg") for p in ("1", "2", "3")
    ]


def test_val_split():
    dataset_path = pd.DataFrame({"a": [0] * 100})
    df = validation_split(dataset_path, 0.2, random_state=0)
    assert len(df[~df["train"]]) == 20
