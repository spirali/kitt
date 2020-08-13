import sys
from pathlib import Path

TEST_DIR = Path(__file__).absolute().parent
ROOT_DIR = TEST_DIR.parent

sys.path.insert(0, str(ROOT_DIR))


def data_path(path: str) -> str:
    return str(TEST_DIR / "data" / path)
