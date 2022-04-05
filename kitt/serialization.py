import dataclasses
import typing

import dacite
from dacite import Config
from yaml import SafeDumper

# Dataclass (de)serialization


def dataclass_to_dict(obj):
    return dataclasses.asdict(obj)


def dataclass_from_dict(cls, data, **config_kwargs):
    if "strict_unions_match" not in config_kwargs:
        config_kwargs["strict_unions_match"] = True

    config = Config(**config_kwargs)

    from dacite.types import is_union

    if is_union(cls):
        from dacite.core import _build_value_for_union

        return _build_value_for_union(cls, data, config)
    return dacite.from_dict(cls, data, config=config)


def tagged_dataclass(cls=None, /, tag_field="type", **kwargs):
    """
    Use this decorator instead of `dataclasses.dataclass` to create a dataclass that contains an
    automatically created tag field containing its name.
    This is useful for creating "ADT-like" dataclasses which can be then serialized and deserialized
    using `dataclass_to_dict`/`dataclass_from_dict`.

    Example:
        ```
        @tagged_dataclass
        class A:
            a: int

        @tagged_dataclass
        class B:
            a: int

        MyAdt = typing.Union[A, B]

        obj = A(a=5)
        serialized = dataclass_to_dict(obj)
        deserialized = dataclass_from_dict(MyAdt, serialized)
        # `deserialized` was correctly deserialized as `A`!
        ```
    """

    def wrap(cls):
        class_name = cls.__name__
        assert tag_field not in cls.__annotations__

        # Set annotation
        cls.__annotations__[tag_field] = typing.Literal[class_name]

        # Set field
        setattr(
            cls,
            tag_field,
            dataclasses.field(
                default=class_name, hash=False, compare=False, repr=False, init=False
            ),
        )

        # Set field setter
        return dataclasses.dataclass(cls, **kwargs)

    if cls is None:
        return wrap
    return wrap(cls)


# YAML serialization


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
