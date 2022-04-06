import dataclasses
from io import StringIO
from typing import Union

import pytest

from kitt.serialization import (
    dataclass_from_dict,
    dataclass_to_dict,
    tagged_dataclass,
    write_yaml,
)


def test_serialize_dataclass_yaml():
    @dataclasses.dataclass
    class Foo:
        a: int = 5
        b: str = "foo"

    @dataclasses.dataclass
    class Bar:
        c: str
        d: Foo

    stream = StringIO()
    obj = Bar(c="c", d=Foo())
    write_yaml(obj, stream)
    assert (
        stream.getvalue()
        == """c: c
d:
  a: 5
  b: foo
"""
    )


def test_tag_dataclass_instance():
    @tagged_dataclass
    class Foo:
        a: int = 5

    foo = Foo()
    assert foo.type == "Foo"


def test_tag_dataclass_class():
    @tagged_dataclass
    class Foo:
        a: int = 5

    assert Foo.type == "Foo"


def test_tag_dataclass_custom_field_name():
    @tagged_dataclass(tag_field="foo")
    class Foo:
        a: int = 5

    foo = Foo()
    assert foo.foo == "Foo"


def test_tagged_dataclass_no_fields():
    @tagged_dataclass
    class Foo:
        pass

    foo = Foo()
    assert dataclass_roundtrip(Foo, foo) == foo


def test_tagged_dataclass_no_fields_with_inheritance():
    @dataclasses.dataclass
    class Base:
        a: int

    @tagged_dataclass
    class Foo(Base):
        pass

    foo = Foo(a=1)
    assert dataclass_roundtrip(Foo, foo) == foo


def test_tagged_dataclass_pass_arguments():
    @tagged_dataclass(frozen=True)
    class Foo:
        a: int = 5

    foo = Foo()
    with pytest.raises(dataclasses.FrozenInstanceError):
        foo.a = 1


def test_tagged_dataclass_change_field():
    @tagged_dataclass
    class Foo:
        a: int = 5

    foo = Foo()
    foo.a = 8
    assert foo.a == 8


def test_tagged_dataclass_with_inheritance():
    @dataclasses.dataclass
    class Base:
        x: str

    @tagged_dataclass
    class Foo(Base):
        a: int = 5

    foo = Foo(x="foo", a=1)
    assert foo.type == "Foo"


def test_nested_tagged_dataclass_from_dict():
    @tagged_dataclass
    class Cls1:
        a: int = 5

    @tagged_dataclass
    class Cls2:
        a: int = 5

    Adt = Union[Cls1, Cls2]

    @dataclasses.dataclass
    class Data:
        data: Adt

    data = Data(data=Cls1(a=1))
    data2 = dataclass_roundtrip(Data, data)
    assert isinstance(data2.data, Cls1)
    assert data == data2

    data = Data(data=Cls2(a=1))
    data2 = dataclass_roundtrip(Data, data)
    assert isinstance(data2.data, Cls2)
    assert data == data2


def test_tagged_dataclass_from_dict():
    @tagged_dataclass
    class Cls1:
        a: int = 5

    @tagged_dataclass
    class Cls2:
        a: int = 5

    Adt = Union[Cls1, Cls2]

    obj = Cls1(a=1)
    obj2 = dataclass_roundtrip(Adt, obj)
    assert obj == obj2

    obj = Cls2(a=1)
    obj2 = dataclass_roundtrip(Adt, obj)
    assert obj == obj2


def dataclass_roundtrip(type, obj):
    serialized = dataclass_to_dict(obj)
    return dataclass_from_dict(type, serialized)
