from typing import Optional

import click

# Prepared CLI options
NumGPUs = click.option("--num-gpus", default=0, type=int)


# CLI types
class Resolution(click.ParamType):
    name = "resolution"

    def convert(self, value: Optional[str], param, ctx):
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and all(isinstance(v, int) for v in value)
        ):
            return value

        try:
            parts = value.split(",")
            if len(parts) != 2:
                raise ValueError()
            return int(parts[0]), int(parts[1])
        except ValueError:
            self.fail("Expected resolution in the format <width>,<height>")


class Separated(click.ParamType):
    name = "separated"

    def __init__(self, separator=","):
        self.separator = separator

    def convert(self, value: Optional[str], param, ctx):
        if value is None:
            return ()

        if isinstance(value, tuple):
            return value

        try:
            return value.split(self.separator)
        except ValueError:
            self.fail(f"Expected values separated by {self.separator}")
