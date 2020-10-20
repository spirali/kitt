import os
import sys

from .utils import get_process_output


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
