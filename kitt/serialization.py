import dataclasses

import dacite


def dataclass_to_dict(obj):
    return dataclasses.asdict(obj)


def dataclass_from_dict(cls, data):
    return dacite.from_dict(cls, data)
