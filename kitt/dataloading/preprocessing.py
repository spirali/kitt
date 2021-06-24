class Preprocessing:
    """
    Represents data preprocessing that can normalize (and potentially also denormalize) some data.
    """

    def normalize(self, data):
        raise NotImplementedError()

    def denormalize(self, data):
        raise NotImplementedError()


class IdentityPreprocessing(Preprocessing):
    """
    Data preprocessing that doesn't change input data in any way.
    """

    def normalize(self, data):
        return data

    def denormalize(self, data):
        return data


class FunctionPreprocessing(Preprocessing):
    """
    Data preprocessing that modifies data using a (de)normalize function.
    """

    def __init__(self, normalize_fn, denormalize_fn=None):
        self.normalize_fn = normalize_fn
        self.denormalize_fn = denormalize_fn

    def normalize(self, data):
        return self.normalize_fn(data)

    def denormalize(self, data):
        if self.denormalize_fn is None:
            raise NotImplementedError()
        return self.denormalize_fn(data)


class ScalePreprocessing(Preprocessing):
    """
    Data preprocessing that scales (multiplies) the data by the given value.
    """

    def __init__(self, scale: float):
        self.scale = scale

    def normalize(self, data):
        return data * self.scale

    def denormalize(self, data):
        return data / self.scale
