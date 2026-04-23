from typer.testing import CliRunner

from clipper import __version__
from clipper.cli import app


def test_version_command_prints_package_version() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout == f"clipper-tool {__version__}\n"
