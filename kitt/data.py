class TrainTestPair:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data


class Data:

    __slots__ = ("name", "inputs", "labels", "predictions")

    def __init__(self, name, inputs=None, labels=None, predictions=None):
        self.name = name
        self.inputs = inputs
        self.labels = labels
        self.predictions = predictions

    def make_il_pair(self):
        return self.inputs, self.labels

    def __getitem__(self, item):
        return Data(
            self.name,
            self.inputs[item] if self.inputs is not None else None,
            self.labels[item] if self.labels is not None else None,
            self.predictions[item] if self.predictions is not None else None,
        )

    def __len__(self):
        if self.inputs is not None:
            return len(self.inputs)
        if self.labels is not None:
            return len(self.labels)
        if self.predictions is not None:
            return len(self.predictions)
        return 0

    def __iter__(self):
        inputs = self.inputs
        labels = self.labels
        predictions = self.predictions

        for i in range(len(self)):
            yield (
                inputs[i] if inputs is not None else None,
                labels[i] if labels is not None else None,
                predictions[i] if predictions is not None else None,
            )
