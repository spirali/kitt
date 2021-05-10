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
    """Return the extension of a file path."""
    return os.path.splitext(path)[1]


def clear_tf_memory():
    from tensorflow import keras

    keras.backend.clear_session()
