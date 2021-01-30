import io

import h5py
from tensorflow import keras
from tensorflow.python.keras.models import load_model, save_model

from .environment import get_arguments


class ModelWrapper:
    def __init__(self):
        self.params = get_arguments(1)
        if "self" in self.params:
            del self.params["self"]


def load_model_from_bytes(data: bytes) -> keras.Model:
    bytes = io.BytesIO(data)
    with h5py.File(bytes, "r") as f:
        return load_model(f)


def save_model_to_bytes(model: keras.Model) -> bytes:
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as f:
        save_model(model, f)
    return buffer.getvalue()
