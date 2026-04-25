import json
from pathlib import Path

import typer
from pydantic import ValidationError

from clipper import __version__
from clipper.adapters.ffmpeg_render import FFmpegRenderError
from clipper.models.job import ClipperJob
from clipper.pipeline.brief import BriefResponseError
from clipper.pipeline.orchestrator import RenderStageError, VALID_STAGES, run_job
from clipper.pipeline.scoring import ScoringResponseError

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Keep the CLI in group mode as the starting shape for a multi-command app."""
    pass


@app.command()
def version() -> None:
    typer.echo(f"clipper-tool {__version__}")


@app.command()
def run(
    job_path: Path,
    stage: str = typer.Option(
        "auto",
        "--stage",
        help="Pipeline stage: mine, brief, review, render, or auto.",
    ),
) -> None:
    if stage not in VALID_STAGES:
        typer.echo(
            f"Invalid --stage {stage!r}. Expected one of: {', '.join(VALID_STAGES)}.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        job_text = job_path.read_text(encoding="utf-8")
    except OSError:
        typer.echo("Unable to read job file.", err=True)
        raise typer.Exit(code=1)

    try:
        payload = json.loads(job_text)
    except json.JSONDecodeError:
        typer.echo("Invalid job file JSON.", err=True)
        raise typer.Exit(code=1)

    try:
        job = ClipperJob.model_validate(payload)
    except ValidationError:
        typer.echo("Invalid job file payload.", err=True)
        raise typer.Exit(code=1)

    try:
        artifact_path = run_job(job, stage=stage)
    except BriefResponseError as exc:
        typer.echo(f"Brief handoff error: {exc}", err=True)
        raise typer.Exit(code=1)
    except ScoringResponseError as exc:
        typer.echo(f"Scoring handoff error: {exc}", err=True)
        raise typer.Exit(code=1)
    except RenderStageError as exc:
        typer.echo(f"Render stage error: {exc}", err=True)
        raise typer.Exit(code=1)
    except FFmpegRenderError as exc:
        typer.echo(f"Render failed: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(str(artifact_path))


if __name__ == "__main__":
    app()
