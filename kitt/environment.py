import dataclasses
import inspect
import os
import re
import sys
import time
from datetime import datetime

from yaml import SafeDumper

from .utils import get_extension, get_process_output


class CustomDumper(SafeDumper):
    def ignore_aliases(self, data):
        return True

    def represent_data(self, data):
        if dataclasses.is_dataclass(data):
            return super().represent_dict(dataclasses.asdict(data))
        return super().represent_data(data)


def write_yaml(object, stream):
    import yaml

    yaml.dump(object, stream, Dumper=CustomDumper)


def write_environment_yaml(path: str, **kwargs):
    """Store information about the environment into the passed YAML file"""
    assert get_extension(path) in (".yml", ".yaml")

    data = get_environment()
    if kwargs:
        data["user_data"] = kwargs

    # move environment to the end of the file
    if "env" in data:
        data["zzz_env"] = data["env"]
        del data["env"]

    with open(path, "w") as f:
        write_yaml(data, f)


def get_environment():
    return {
        "args": " ".join([sys.executable] + sys.argv),
        "git": get_git_info(),
        "packages": get_packages_info(),
        "env": os.environ.copy(),
        "time": {"unix": time.time(), "date": str(datetime.now())},
    }


def get_git_info():
    changed = get_process_output(["git", "diff", "--name-status", "HEAD"]).splitlines()
    changed = [line.strip() for line in changed]

    return {
        "branch": get_process_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "sha": get_process_output(["git", "rev-parse", "HEAD"]),
        "changes": changed,
    }


def get_packages_info():
    package_regex = re.compile(r"^(.*)==(.*)$")
    packages = {}

    for line in get_process_output(["pip", "freeze"]).splitlines():
        match = package_regex.match(line.strip())
        if match:
            packages[match.group(1)] = match.group(2)

    return packages


def get_arguments(index=0):
    """Yields frames and arguments of functions in the current call stack, starting from the
    selected index.

    :param index: 0 - calling function, 1 - parent of calling function etc.
    """
    frame = inspect.currentframe()
    frames = inspect.getouterframes(frame)
    if len(frames) < index + 2:
        return None
    for parent_frame in frames[index + 1 :]:
        args = inspect.getargvalues(parent_frame.frame)
        context = {}

        def assign(names):
            for name in names:
                if name in args.locals:
                    context[name] = args.locals[name]

        assign(args.args)
        assign([args.keywords])
        assign([args.varargs])
        yield (parent_frame, context)


def get_constructor_arguments():
    arguments = {}

    for (frame, args) in get_arguments(1):
        if frame.function == "__init__":
            arguments.update(args)
        else:
            break

    if "self" in arguments:
        del arguments["self"]
    return arguments
