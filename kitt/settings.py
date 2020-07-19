import inspect
from collections import namedtuple

Entry = namedtuple("Entry", ["name", "location"])


class Settings:
    def __init__(self):
        self.functions = {}

    def register(self, group, name, fn, is_buildin):
        if is_buildin:
            location = None
        else:
            location = "{}:{}".format(
                inspect.getsourcefile(fn), inspect.getsourcelines(fn)[1]
            )

        if group not in self.functions:
            self.functions[group] = []
        self.functions[group].append(Entry(name, location))
