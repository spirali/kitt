import collections

from kitt.data import Data

ModelInfo = collections.namedtuple("ModelInfo", ["name", "model"])


def _make_data_info(name, data):
    assert isinstance(name, str)
    labels = None
    predictions = None
    if len(data) == 1:
        inputs = data
    elif len(data) == 2:
        inputs, labels = data
    elif len(data) == 3:
        inputs, labels, predictions = data
    else:
        raise Exception("Invalid data shape")
    return Data(name, inputs, labels, predictions)


class State:
    def __init__(self):
        self.models = []
        self.train_data = []
        self.test_data = []

    def add_model(self, name, model):
        assert isinstance(name, str)
        self.models.append(ModelInfo(name, model))

    def add_train_data(self, name, data):
        self.train_data.append(_make_data_info(name, data))

    def add_test_data(self, name, data):
        self.test_data.append(_make_data_info(name, data))
