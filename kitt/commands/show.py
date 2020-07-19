import click

from ..decorators import command
from ..globals import get_global_settings


@command(_is_buildin=True, group="Inspect")
def commands(_state):
    settings = get_global_settings()

    for group_name, group in sorted(settings.functions.items()):
        click.secho("{}:".format(group_name), fg="blue", bold=True)
        for entry in sorted(group, key=lambda e: e.name):
            location = entry.location
            click.echo("\t{} ".format(entry.name), nl=False)
            if location:
                click.secho("[{}]".format(location), fg="cyan")
            else:
                click.secho("<build-in>", fg="magenta")
        click.echo()


@command(_is_buildin=True, group="Inspect")
def show_state(state):
    if not state.models and not state.test_data and not state.train_data:
        click.secho("State is empty", fg="red")

    if state.models:
        click.secho("{}:".format("Models"), fg="blue", bold=True)
        for info in state.models:
            click.echo("\t{}".format(info.name))
        click.echo()

    if state.train_data:
        click.secho("{}:".format("Train data"), fg="blue", bold=True)
        for info in state.train_data:
            click.echo("\t{}".format(info.name))
        click.echo()

    if state.test_data:
        click.secho("{}:".format("Test data"), fg="blue", bold=True)
        for info in state.test_data:
            click.echo("\t{}".format(info.name))
        click.echo()


@command(_is_buildin=True, group="Inspect")
def show_models(state):
    for info in state.models:
        click.secho("--- Model '{}' ---".format(info.name), fg="blue", bold=True)
        info.model.summary()
        click.echo()
