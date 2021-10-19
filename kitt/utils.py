import logging
import os
import subprocess
from typing import Union


def get_process_output(args) -> Union[str, None]:
    try:
        result = subprocess.run(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return result.stdout.decode().strip()
    except Exception as e:
        logging.error(e)
        return None


def get_extension(path: str) -> str:
    """
    Return the extension of a file path.

    The extension will include the dot.
    """
    return os.path.splitext(path)[1]
