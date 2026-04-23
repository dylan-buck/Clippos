import json

import pytest
from typer.testing import CliRunner

from clipper import __version__
from clipper.adapters.ffmpeg_render import FFmpegRenderError
from clipper.cli import app
from clipper.pipeline.orchestrator import RenderStageError
from clipper.pipeline.scoring import ScoringResponseError


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def sample_job_payload(tmp_path) -> dict:
    video_path = tmp_path / "input.mp4"
    return {
        "video_path": str(video_path),
        "output_dir": str(tmp_path / "out"),
    }


def test_version_command_prints_package_version() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout == f"clipper-tool {__version__}\n"


def test_run_command_prints_manifest_path(
    cli_runner: CliRunner,
    tmp_path,
    sample_job_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(sample_job_payload), encoding="utf-8")
    monkeypatch.setattr(
        "clipper.cli.run_job",
        lambda _job, *, stage: tmp_path / "review-manifest.json",
    )

    result = cli_runner.invoke(app, ["run", str(job_path)])

    assert result.exit_code == 0
    assert "review-manifest.json" in result.stdout


def test_run_command_forwards_stage_flag(
    cli_runner: CliRunner,
    tmp_path,
    sample_job_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_run_job(_job, *, stage: str):
        captured["stage"] = stage
        return tmp_path / "scoring-request.json"

    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(sample_job_payload), encoding="utf-8")
    monkeypatch.setattr("clipper.cli.run_job", fake_run_job)

    result = cli_runner.invoke(app, ["run", str(job_path), "--stage", "mine"])

    assert result.exit_code == 0
    assert captured == {"stage": "mine"}


def test_run_command_defaults_stage_to_auto(
    cli_runner: CliRunner,
    tmp_path,
    sample_job_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_run_job(_job, *, stage: str):
        captured["stage"] = stage
        return tmp_path / "review-manifest.json"

    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(sample_job_payload), encoding="utf-8")
    monkeypatch.setattr("clipper.cli.run_job", fake_run_job)

    result = cli_runner.invoke(app, ["run", str(job_path)])

    assert result.exit_code == 0
    assert captured == {"stage": "auto"}


def test_run_command_rejects_unknown_stage(
    cli_runner: CliRunner, tmp_path, sample_job_payload: dict
) -> None:
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(sample_job_payload), encoding="utf-8")

    result = cli_runner.invoke(app, ["run", str(job_path), "--stage", "publish"])

    assert result.exit_code == 1
    assert "Invalid --stage" in result.output


def test_run_command_forwards_render_stage(
    cli_runner: CliRunner,
    tmp_path,
    sample_job_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_run_job(_job, *, stage: str):
        captured["stage"] = stage
        return tmp_path / "render-report.json"

    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(sample_job_payload), encoding="utf-8")
    monkeypatch.setattr("clipper.cli.run_job", fake_run_job)

    result = cli_runner.invoke(app, ["run", str(job_path), "--stage", "render"])

    assert result.exit_code == 0
    assert captured == {"stage": "render"}


def test_run_command_surfaces_render_stage_errors(
    cli_runner: CliRunner,
    tmp_path,
    sample_job_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(sample_job_payload), encoding="utf-8")

    def exploding_run(_job, *, stage: str):
        raise RenderStageError("no review manifest")

    monkeypatch.setattr("clipper.cli.run_job", exploding_run)

    result = cli_runner.invoke(app, ["run", str(job_path), "--stage", "render"])

    assert result.exit_code == 1
    assert "Render stage error" in result.output
    assert "no review manifest" in result.output


def test_run_command_surfaces_ffmpeg_render_errors(
    cli_runner: CliRunner,
    tmp_path,
    sample_job_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(sample_job_payload), encoding="utf-8")

    def exploding_run(_job, *, stage: str):
        raise FFmpegRenderError("ffmpeg exited 1")

    monkeypatch.setattr("clipper.cli.run_job", exploding_run)

    result = cli_runner.invoke(app, ["run", str(job_path), "--stage", "render"])

    assert result.exit_code == 1
    assert "Render failed" in result.output
    assert "ffmpeg exited 1" in result.output


def test_run_command_surfaces_scoring_response_errors(
    cli_runner: CliRunner,
    tmp_path,
    sample_job_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(sample_job_payload), encoding="utf-8")

    def exploding_run(_job, *, stage: str):
        raise ScoringResponseError("response is malformed")

    monkeypatch.setattr("clipper.cli.run_job", exploding_run)

    result = cli_runner.invoke(app, ["run", str(job_path)])

    assert result.exit_code == 1
    assert "Scoring handoff error" in result.output
    assert "response is malformed" in result.output


def test_run_command_reports_invalid_json(cli_runner: CliRunner, tmp_path) -> None:
    job_path = tmp_path / "job.json"
    job_path.write_text("{invalid", encoding="utf-8")

    result = cli_runner.invoke(app, ["run", str(job_path)])

    assert result.exit_code == 1
    assert result.output == "Invalid job file JSON.\n"


def test_run_command_reports_missing_job_file(cli_runner: CliRunner, tmp_path) -> None:
    job_path = tmp_path / "missing.json"

    result = cli_runner.invoke(app, ["run", str(job_path)])

    assert result.exit_code == 1
    assert result.output == "Unable to read job file.\n"


def test_run_command_reports_invalid_job_payload(
    cli_runner: CliRunner, tmp_path
) -> None:
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps({"video_path": "/tmp/input.mp4"}), encoding="utf-8")

    result = cli_runner.invoke(app, ["run", str(job_path)])

    assert result.exit_code == 1
    assert result.output == "Invalid job file payload.\n"
