import logging
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
