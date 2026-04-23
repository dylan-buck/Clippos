import json
from pathlib import Path

import pytest

from clipper.models.job import ClipperJob
from clipper.pipeline.orchestrator import run_job
from clipper.pipeline.scoring import (
    SCORING_REQUEST_FILENAME,
    ScoringResponseError,
)

MOCK_TRANSCRIPT = {
    "segments": [
        {
            "speaker": "speaker_1",
            "start_seconds": 0.0,
            "end_seconds": 8.0,
            "text": "Here's the thing: this crazy secret will ban your favorite snack.",
            "words": [],
        },
        {
            "speaker": "speaker_1",
            "start_seconds": 8.0,
            "end_seconds": 18.0,
            "text": "Turns out the outcome was wild and the room erupted instantly.",
            "words": [],
        },
    ]
}

MOCK_VISION = {
    "frames": [
        {
            "timestamp_seconds": 1.0,
            "motion_score": 0.82,
            "shot_change": True,
            "primary_face": None,
        },
        {
            "timestamp_seconds": 10.0,
            "motion_score": 0.74,
            "shot_change": False,
            "primary_face": None,
        },
    ]
}


def _patch_pipeline_stages(
    monkeypatch: pytest.MonkeyPatch, observed_paths: list[Path]
) -> None:
    monkeypatch.setattr(
        "clipper.pipeline.orchestrator.probe_video",
        lambda path: (
            observed_paths.append(path)
            or {
                "duration_seconds": 120.0,
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "audio_sample_rate": 48000,
            }
        ),
    )
    monkeypatch.setattr(
        "clipper.pipeline.orchestrator.transcribe_video",
        lambda path, workspace: observed_paths.append(path) or MOCK_TRANSCRIPT,
    )
    monkeypatch.setattr(
        "clipper.pipeline.orchestrator.analyze_video",
        lambda path, workspace: observed_paths.append(path) or MOCK_VISION,
    )


def test_run_job_auto_stage_returns_review_manifest_when_scorer_resolves(
    sample_job: ClipperJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(sample_job.video_path.parent.resolve())
    expected_video_path = (Path.cwd() / sample_job.video_path).resolve(strict=False)
    observed_paths: list[Path] = []
    _patch_pipeline_stages(monkeypatch, observed_paths)

    mock_model_scores = [
        {
            "clip_id": "clip-000",
            "title": "Secret payoff",
            "hook": "Watch this reveal",
            "reasons": ["shareability", "payoff"],
            "final_score": 0.88,
        }
    ]
    monkeypatch.setattr(
        "clipper.pipeline.orchestrator.score_shortlist",
        lambda _workspace: mock_model_scores,
    )

    manifest_path = run_job(sample_job)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path.name == "review-manifest.json"
    assert manifest_path.exists()
    assert observed_paths == [
        expected_video_path,
        expected_video_path,
        expected_video_path,
    ]
    assert manifest["video_path"] == str(expected_video_path)
    assert manifest["candidates"][0]["title"] == "Secret payoff"
    assert manifest["candidates"][0]["hook"] == "Watch this reveal"
    assert manifest["candidates"][0]["reasons"] == ["shareability", "payoff"]
    assert manifest["candidates"][0]["score"] == 0.88


def test_run_job_auto_stage_returns_scoring_request_when_no_scores_available(
    sample_job: ClipperJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(sample_job.video_path.parent.resolve())
    observed_paths: list[Path] = []
    _patch_pipeline_stages(monkeypatch, observed_paths)

    artifact_path = run_job(sample_job)

    assert artifact_path.name == SCORING_REQUEST_FILENAME
    assert artifact_path.exists()


def test_run_job_mine_stage_writes_scoring_request_and_skips_review(
    sample_job: ClipperJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(sample_job.video_path.parent.resolve())
    observed_paths: list[Path] = []
    _patch_pipeline_stages(monkeypatch, observed_paths)

    score_shortlist_calls: list[Path] = []

    def _explode(workspace: Path) -> list[dict]:
        score_shortlist_calls.append(workspace)
        raise AssertionError("score_shortlist should not run during mine-only stage")

    monkeypatch.setattr("clipper.pipeline.orchestrator.score_shortlist", _explode)

    artifact_path = run_job(sample_job, stage="mine")

    assert artifact_path.name == SCORING_REQUEST_FILENAME
    assert artifact_path.exists()
    assert score_shortlist_calls == []


def test_run_job_review_stage_raises_when_no_scoring_context_exists(
    sample_job: ClipperJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(sample_job.video_path.parent.resolve())
    monkeypatch.setattr(
        "clipper.pipeline.orchestrator.probe_video",
        lambda _path: {
            "duration_seconds": 120.0,
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "audio_sample_rate": 48000,
        },
    )

    with pytest.raises(ScoringResponseError):
        run_job(sample_job, stage="review")


def test_run_job_review_stage_builds_manifest_from_existing_scoring_artifacts(
    sample_job: ClipperJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(sample_job.video_path.parent.resolve())
    observed_paths: list[Path] = []
    _patch_pipeline_stages(monkeypatch, observed_paths)

    # First run in mine stage to lay down the scoring-request artifact.
    run_job(sample_job, stage="mine")

    mock_model_scores = [
        {
            "clip_id": "clip-000",
            "title": "Review-stage title",
            "hook": "Review-stage hook",
            "reasons": ["payoff"],
            "final_score": 0.72,
        }
    ]
    monkeypatch.setattr(
        "clipper.pipeline.orchestrator.score_shortlist",
        lambda _workspace: mock_model_scores,
    )

    manifest_path = run_job(sample_job, stage="review")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path.name == "review-manifest.json"
    assert manifest["candidates"][0]["title"] == "Review-stage title"
    assert manifest["candidates"][0]["score"] == 0.72


def test_run_job_rejects_unknown_stage(sample_job: ClipperJob) -> None:
    with pytest.raises(ValueError):
        run_job(sample_job, stage="render")  # type: ignore[arg-type]
