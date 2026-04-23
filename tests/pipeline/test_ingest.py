from pathlib import Path

import pytest
from pydantic import ValidationError

from clipper.models.job import ClipperJob
from clipper.pipeline.ingest import ingest_job


def test_ingest_job_builds_workspace_and_probe_result(tmp_path: Path) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")
    job = ClipperJob(video_path=video, output_dir=tmp_path / "out")

    result = ingest_job(
        job,
        probe_data={
            "duration_seconds": 120.0,
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "audio_sample_rate": 48000,
        },
    )

    assert result.workspace_dir == job.output_dir / "jobs" / result.job_id
    assert result.probe.duration_seconds == 120.0


def test_ingest_job_uses_stable_job_id(tmp_path: Path) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")
    job = ClipperJob(video_path=video, output_dir=tmp_path / "out")

    one = ingest_job(
        job,
        probe_data={
            "duration_seconds": 1.0,
            "width": 1,
            "height": 1,
            "fps": 1.0,
            "audio_sample_rate": 16000,
        },
    )
    two = ingest_job(
        job,
        probe_data={
            "duration_seconds": 1.0,
            "width": 1,
            "height": 1,
            "fps": 1.0,
            "audio_sample_rate": 16000,
        },
    )

    assert one.job_id == two.job_id


def test_ingest_job_normalizes_raw_ffprobe_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")
    job = ClipperJob(video_path=video, output_dir=tmp_path / "out")

    monkeypatch.setattr(
        "clipper.pipeline.ingest.probe_media",
        lambda _path: {
            "format": {"duration": "120.0"},
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "avg_frame_rate": "30000/1000",
                },
                {
                    "codec_type": "audio",
                    "sample_rate": "48000",
                },
            ],
        },
    )

    result = ingest_job(job)

    assert result.probe.model_dump() == {
        "duration_seconds": 120.0,
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "audio_sample_rate": 48000,
    }


def test_ingest_job_does_not_create_workspace_on_probe_validation_failure(
    tmp_path: Path,
) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")
    job = ClipperJob(video_path=video, output_dir=tmp_path / "out")

    with pytest.raises(ValidationError):
        ingest_job(
            job,
            probe_data={
                "duration_seconds": 0.0,
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "audio_sample_rate": 48000,
            },
        )

    assert not (job.output_dir / "jobs").exists()


def test_ingest_job_uses_same_job_id_for_equivalent_path_spellings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")

    monkeypatch.chdir(tmp_path)

    absolute_job = ClipperJob(video_path=video, output_dir=tmp_path / "out")
    relative_job = ClipperJob(video_path=Path("input.mp4"), output_dir=tmp_path / "out")

    absolute_result = ingest_job(
        absolute_job,
        probe_data={
            "duration_seconds": 1.0,
            "width": 1,
            "height": 1,
            "fps": 1.0,
            "audio_sample_rate": 16000,
        },
    )
    relative_result = ingest_job(
        relative_job,
        probe_data={
            "duration_seconds": 1.0,
            "width": 1,
            "height": 1,
            "fps": 1.0,
            "audio_sample_rate": 16000,
        },
    )

    assert absolute_result.job_id == relative_result.job_id


def test_ingest_job_uses_canonical_path_for_live_probe_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    video = home_dir / "input.mp4"
    video.write_bytes(b"fake")

    monkeypatch.setenv("HOME", str(home_dir))

    observed_paths: list[Path] = []

    def fake_probe_media(video_path: Path) -> dict:
        observed_paths.append(video_path)
        return {
            "format": {"duration": "1.0"},
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1,
                    "height": 1,
                    "avg_frame_rate": "1/1",
                },
                {"codec_type": "audio", "sample_rate": "16000"},
            ],
        }

    monkeypatch.setattr("clipper.pipeline.ingest.probe_media", fake_probe_media)

    job = ClipperJob(video_path=Path("~/input.mp4"), output_dir=tmp_path / "out")

    ingest_job(job)

    assert observed_paths == [video]
