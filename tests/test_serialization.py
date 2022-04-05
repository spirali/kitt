import dataclasses
from io import StringIO

from kitt.serialization import write_yaml


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
    obj = Bar(
        c="c",
        d=Foo()
    )
    write_yaml(obj, stream)
    assert stream.getvalue() == """c: c
d:
  a: 5
  b: foo
"""
