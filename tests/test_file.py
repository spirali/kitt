from conftest import data_path

from kitt.files import iterate_directories, iterate_files


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
