from .environment import get_arguments


class ModelWrapper:
    def __init__(self):
        self.params = get_arguments(1)
        if "self" in self.params:
            del self.params["self"]
