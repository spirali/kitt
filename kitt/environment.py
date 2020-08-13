import logging
import os
import subprocess
import sys
from typing import Union


def get_environment():
    return {
        "args": " ".join([sys.executable] + sys.argv),
        "git": get_git_info(),
        "env": os.environ.copy(),
    }


def get_git_info():
    return {
        "branch": get_process_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "sha": get_process_output(["git", "rev-parse", "HEAD"]),
    }


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
