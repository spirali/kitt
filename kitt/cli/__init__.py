import click

# Prepared CLI options
NumGPUs = click.option("--num-gpus", default=0, type=int)


# CLI types
class Resolution(click.ParamType):
    name = "resolution"

    def convert(self, value, param, ctx):
        try:
            parts = value.split(",")
            if len(parts) != 2:
                raise ValueError()
            return int(parts[0]), int(parts[1])
        except ValueError:
            self.fail("Expected resolution in the format <width>,<height>")
