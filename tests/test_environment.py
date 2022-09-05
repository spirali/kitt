import os
from dataclasses import dataclass
from typing import List

from kitt.environment import get_git_info, write_environment_yaml


def test_serialize_dataclass(tmpdir):
    @dataclass
    class Foo:
        a: int
        b: List[int]

    foo = Foo(5, [1, 2, 3])
    write_environment_yaml(os.path.join(tmpdir, "params.yaml"), foo=foo)


def test_get_git_info(tmp_path):
    original_dir = os.getcwd()
    try:
        os.chdir(tmp_path)
        get_git_info()
    finally:
        os.chdir(original_dir)
