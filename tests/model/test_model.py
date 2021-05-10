import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense

from kitt.dataloading import BatchLoader, KerasSequence, ListDataLoader
from kitt.dataloading.preprocessing import Preprocessing, ScalePreprocessing
from kitt.model import ModelWrapper


def test_model_map_loader():
    a = [np.array([v]) for v in range(5)]
    b = [np.array([v]) for v in range(5, 10)]
    loader = ListDataLoader(list(zip(a, b)))

    class Model(ModelWrapper):
        def input_preprocessing(self) -> Preprocessing:
            return ScalePreprocessing(2.0)

        def output_preprocessing(self) -> Preprocessing:
            return ScalePreprocessing(3.0)

    model = Model()
    model_loader = model.map_loader(loader)
    data = list(model_loader)
    a_mapped = [v * 2 for v in a]
    b_mapped = [v * 3 for v in b]
    assert data == list(zip(a_mapped, b_mapped))


def test_model_parallel_train():
    a = [np.array([v]) for v in range(5)]
    b = [np.array([v]) for v in range(5, 10)]
    loader = ListDataLoader(list(zip(a, b)))
    loader = KerasSequence(BatchLoader(loader, 2))

    class Model(ModelWrapper):
        def input_preprocessing(self) -> Preprocessing:
            return ScalePreprocessing(2.0)

        def output_preprocessing(self) -> Preprocessing:
            return ScalePreprocessing(3.0)

        def build_network(self) -> keras.Model:
            return keras.Sequential([Dense(50), Dense(2)])

        def compile(self, model: keras.Model):
            model.compile(optimizer="adam", loss="mse")

    model = Model()
    network = model.build()
    network.fit(loader, epochs=2, workers=2, use_multiprocessing=True)
