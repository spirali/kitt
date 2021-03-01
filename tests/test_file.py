from kitt.files import iterate_files, iterate_directories
from tests.conftest import data_dir_path


def test_iterate_files():
    assert len(list(iterate_files(data_dir_path(), "jpeg"))) == 4
    assert len(list(iterate_files(data_dir_path(), ".jpeg"))) == 4
    assert len(list(iterate_files(data_dir_path(), "jpeg", "e"))) == 1
    assert len(list(iterate_files(data_dir_path(), "png"))) == 0
    assert len(list(iterate_files(data_dir_path(), "jpeg", "a"))) == 0


def test_iterate_directories():
    files = list(iterate_directories((data_dir_path(), data_dir_path()), "jpeg"))
    assert len(files) == 8
    files = list(iterate_directories((data_dir_path(), data_dir_path()), "jpeg", "e"))
    assert len(files) == 2
    files = list(iterate_directories((data_dir_path(), data_dir_path()), "png"))
    assert len(files) == 0
