import logging

import click
import coloredlogs

from .state import State

logger = logging.getLogger("kitt")

pass_state = click.make_pass_decorator(State, ensure=True)


@click.group(chain=True)
@click.option(
    "--log",
    default="info",
    type=click.Choice(["critical", "error", "warning", "info", "debug"]),
)
@pass_state  # Must be there for top-level initialization
def cli_main(_state, log):
    levels = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }
    coloredlogs.install(
        level=levels[log],
        fmt="%(asctime)s %(hostname)s %(name)s %(levelname)s %(message)s",
    )

    # TODO: Add option to disable this
    from silence_tensorflow import silence_tensorflow

    silence_tensorflow()
