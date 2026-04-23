import typer

from clipper import __version__

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Keep the CLI in group mode as the starting shape for a multi-command app."""
    pass


@app.command()
def version() -> None:
    typer.echo(f"clipper-tool {__version__}")
