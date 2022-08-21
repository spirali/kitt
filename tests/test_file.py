from kitt.files import iterate_directories, iterate_files, iterate_files_from

from .conftest import data_path


def test_iterate_files():
    directory = data_path("files/d1")
    assert len(list(iterate_files(directory, "foo"))) == 3
    assert len(list(iterate_files(directory, ".foo"))) == 3
    assert len(list(iterate_files(directory, "foo", "e"))) == 2
    assert len(list(iterate_files(directory, "foo", "a"))) == 1
    assert len(list(iterate_files(directory, "bar"))) == 0


def test_iterate_directories():
    directory1 = data_path("files/d1")
    directory2 = data_path("files/d2")

    files = list(iterate_directories((directory1, directory2), "foo"))
    assert len(files) == 5
    files = list(iterate_directories((directory1, directory2), "foo", "e"))
    assert len(files) == 3
    files = list(iterate_directories((directory1, directory2), "baz"))
    assert len(files) == 0


def test_iterate_files_from():
    assert list(iterate_files_from(data_path("files/d1/a1.foo"))) == [
        data_path("files/d1/a1.foo")
    ]
    assert list(iterate_files_from(data_path("files"))) == [
        data_path("files/d1/a1.foo"),
        data_path("files/d1/e1.foo"),
        data_path("files/d1/e2.foo"),
        data_path("files/d2/a1.foo"),
        data_path("files/d2/e2.foo"),
    ]
