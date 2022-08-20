from typing import Callable


def named(fn: Callable, name: str) -> Callable:
    """
    Returns `fn` named with the given `name`.
    """

    def fun(*args, **kwargs):
        return fn(*args, **kwargs)

    fun.__name__ = name
    return fun
