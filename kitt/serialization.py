import dataclasses

import dacite
from yaml import SafeDumper


def dataclass_to_dict(obj):
    return dataclasses.asdict(obj)


def dataclass_from_dict(cls, data, **kwargs):
    return dacite.from_dict(cls, data, **kwargs)


class DataclassYamlDumper(SafeDumper):
    def ignore_aliases(self, data):
        return True

    def represent_data(self, data):
        if dataclasses.is_dataclass(data):
            return super().represent_dict(dataclass_to_dict(data))
        return super().represent_data(data)


def write_yaml(object, stream):
    """
    Writes an object into the output stream as YAML.
    Serializes dataclasses as dictionaries.
    """
    import yaml

    yaml.dump(object, stream, Dumper=DataclassYamlDumper)
