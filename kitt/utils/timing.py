import contextlib
import logging
import time


@contextlib.contextmanager
def time_block(name="timer", enabled=True):
    """
    Calculates the duration of the block in which this context manager is active.
    You can use `enabled` to turn off the measurement to without removing the indented block.

    Usage:
    ```python
    with time_block("preprocess"):
        result = preprocessing(data)
    ```
    """
    start = time.time()
    try:
        yield
    finally:
        if enabled:
            duration = time.time() - start
            logging.info(f"{name}: {duration:.4f}s")
