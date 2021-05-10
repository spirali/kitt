class Preprocessing:
    def normalize(self, data):
        raise NotImplementedError()

    def denormalize(self, data):
        raise NotImplementedError()


class IdentityPreprocessing(Preprocessing):
    def normalize(self, data):
        return data

    def denormalize(self, data):
        return data


class FunctionPreprocessing(Preprocessing):
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
    def __init__(self, scale: float):
        self.scale = scale

    def normalize(self, data):
        return data * self.scale

    def denormalize(self, data):
        return data / self.scale
