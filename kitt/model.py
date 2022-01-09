import io
from pathlib import Path
from typing import Union

import h5py
from tensorflow import keras
from tensorflow.keras.models import load_model, save_model

from .dataloading import DataLoader, MappingLoader
from .dataloading.mapping import create_tuple_mapper
from .dataloading.preprocessing import IdentityPreprocessing, Preprocessing
from .environment import get_constructor_arguments


class ModelWrapper:
    def __init__(self):
        self.params = get_constructor_arguments()

    def input_preprocessing(self) -> Preprocessing:
        return IdentityPreprocessing()

    def output_preprocessing(self) -> Preprocessing:
        return IdentityPreprocessing()

    def map_loader(self, loader: DataLoader) -> DataLoader:
        input_preprocess = self.input_preprocessing()
        output_preprocess = self.output_preprocessing()

        return MappingLoader(
            loader,
            create_tuple_mapper(
                input_preprocess.normalize, output_preprocess.normalize
            ),
        )

    def restore_from_checkpoint(self, file: Union[str, bytes]) -> keras.Model:
        if isinstance(file, (str, Path)):
            model = load_model(file, custom_objects=self.get_custom_objects())
        elif isinstance(file, (bytes, bytearray)):
            model = load_model_from_bytes(
                file, custom_objects=self.get_custom_objects()
            )
        else:
            raise Exception(f"Invalid path type {type(file)}")
        self.sanity_check_checkpoint(model)
        self.compile(model)
        return model

    def build(self) -> keras.Model:
        model = self.build_network()
        self.compile(model)
        return model

    def build_network(self) -> keras.Model:
        raise NotImplementedError()

    def compile(self, model: keras.Model):
        raise NotImplementedError()

    def get_custom_objects(self):
        raise NotImplementedError()

    def sanity_check_checkpoint(self, model: keras.Model):
        pass


def load_model_from_bytes(data: bytes, **load_kwargs) -> keras.Model:
    bytes = io.BytesIO(data)
    with h5py.File(bytes, "r") as f:
        return load_model(f, **load_kwargs)


def save_model_to_bytes(model: keras.Model, **save_kwargs) -> bytes:
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as f:
        save_model(model, f, **save_kwargs)
    return buffer.getvalue()
