import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from clipper.models.job import ClipperJob


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def sample_job_payload() -> dict:
    return json.loads((FIXTURES_DIR / "sample_job.json").read_text(encoding="utf-8"))


@pytest.fixture
def sample_transcript_payload() -> dict:
    return json.loads((FIXTURES_DIR / "sample_transcript.json").read_text(encoding="utf-8"))


@pytest.fixture
def sample_face_payload() -> dict:
    return json.loads((FIXTURES_DIR / "sample_faces.json").read_text(encoding="utf-8"))


@pytest.fixture
def sample_job(tmp_path: Path, sample_job_payload: dict) -> ClipperJob:
    video_path = tmp_path / Path(sample_job_payload["video_path"]).name
    output_dir = tmp_path / Path(sample_job_payload["output_dir"]).name
    video_path.write_bytes(b"fake")
    output_dir.mkdir()

    job = ClipperJob(
        **{
            **sample_job_payload,
            "video_path": video_path,
            "output_dir": output_dir,
        }
    )
    return job
