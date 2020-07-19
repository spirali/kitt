import functools

import click

from .cli import cli_main, pass_state
from .data import TrainTestPair
from .globals import get_global_settings
from .logger import logger


def _extract_params(obj):
    try:
        params = obj.__click_params__
        del obj.__click_params__
        params.reverse()
        return params
    except AttributeError:
        return []


def _model_wrapper(__fn, __name, __state, *args, **kwargs):
    logger.info("Model command '{}'".format(__name))
    logger.debug("args = %s; kwargs = %s", args, kwargs)
    result = __fn(*args, **kwargs)
    if isinstance(result, tuple):
        __name, model = result
        logger.info("Model created under name '{}'".format(__name))
    else:
        model = result
    __state.add_model(__name, model)


def _loader_wrapper(__fn, __name, __state, *args, **kwargs):
    logger.info("Loading command '{}'".format(__name))
    logger.debug("args = %s; kwargs = %s", args, kwargs)
    result = __fn(*args, **kwargs)
    if isinstance(result, TrainTestPair):
        train = result.train_data
        test = result.test_data
    else:
        train = result
        test = None

    if train:
        __state.add_train_data(__name, train)
    if test:
        __state.add_test_data(__name, test)


def _command_wrapper(__fn, __name, *args, **kwargs):
    logger.info("Command '{}'".format(__name))
    logger.debug("args = %s; kwargs = %s", args, kwargs)
    __fn(*args, **kwargs)


def _cmd_builder(fn, wrapper, group, name, is_buildin, params=None):
    settings = get_global_settings()
    command_name = name or fn.__name__
    if params is None:
        params = []
    params += _extract_params(fn)
    callback = functools.partial(wrapper, fn, command_name)
    callback = pass_state(callback)
    cmd = click.Command(command_name, callback=callback, params=params)
    cli_main.add_command(cmd)
    settings.register(group, command_name, fn, is_buildin)
    return cmd


def model(*, name=None, _is_buildin=False):
    def _helper(fn):
        return _cmd_builder(fn, _model_wrapper, "Models", name, _is_buildin)

    return _helper


def command(*, name=None, group="User defined", _is_buildin=False):
    def _helper(fn):
        return _cmd_builder(fn, _command_wrapper, group, name, _is_buildin)

    return _helper


def loader(*, name=None, group="Loaders", _is_buildin=False):
    def _helper(fn):
        return _cmd_builder(fn, _loader_wrapper, group, name, _is_buildin)

    return _helper


def _pyplot_wrapper(__fn, __name, __state, show, *args, **kwargs):
    import matplotlib.pyplot as plt

    plt.close()
    logger.info("Pyplot command '{}'".format(__name))
    logger.debug("args = %s; kwargs = %s", args, kwargs)

    data = __state.train_data[-1]
    __fn(data, *args, **kwargs)
    if show:
        plt.show()


def pyplot_command(*, name=None, group="Pyplot", _is_buildin=False):
    def _helper(fn):
        params = [click.Option(("--show",), is_flag=True)]
        return _cmd_builder(fn, _pyplot_wrapper, group, name, _is_buildin, params)

    return _helper
