from click.testing import CliRunner
from kitt.commands.show import commands


def test_commands():
    runner = CliRunner()
    result = runner.invoke(commands)
    assert result.exit_code == 0
    assert result.output == "TODO\n"
