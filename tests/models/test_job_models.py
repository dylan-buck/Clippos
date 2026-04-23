from pathlib import Path

import pytest
from pydantic import ValidationError

from clipper.models.analysis import MediaProbe
from clipper.models.candidate import CandidateClip
from clipper.models.job import ClipperJob, OutputProfile
from clipper.models.render import RenderManifest
from clipper.models.review import ReviewManifest


def test_job_defaults_include_all_output_ratios() -> None:
    job = ClipperJob.model_validate(
        {"video_path": "/tmp/input.mp4", "output_dir": "/tmp/out"}
    )
    assert job.output_profile.ratios == ["9:16", "1:1", "16:9"]


def test_job_requires_existing_review_gate() -> None:
    job = ClipperJob.model_validate(
        {"video_path": "/tmp/input.mp4", "output_dir": "/tmp/out"}
    )
    assert job.review_required is True


def test_output_profile_uses_hook_default_caption_preset() -> None:
    profile = OutputProfile()

    assert profile.caption_preset == "hook-default"


def test_shared_manifests_round_trip_paths_and_candidates() -> None:
    candidate = CandidateClip(
        clip_id="clip-001",
        start_seconds=12.5,
        end_seconds=25.0,
        score=0.91,
        reasons=["clear hook", "speaker emphasis"],
        spike_categories=["hook", "emotion"],
    )

    review_manifest = ReviewManifest(
        job_id="job-123",
        video_path=Path("/tmp/input.mp4"),
        candidates=[candidate],
    )
    render_manifest = RenderManifest(
        clip_id="clip-001",
        approved=True,
        outputs={"9:16": Path("/tmp/out/clip-001-9x16.mp4")},
    )

    assert review_manifest.candidates[0].clip_id == "clip-001"
    assert render_manifest.outputs["9:16"] == Path("/tmp/out/clip-001-9x16.mp4")


def test_media_probe_captures_core_probe_fields() -> None:
    probe = MediaProbe(
        duration_seconds=120.0,
        width=1920,
        height=1080,
        fps=29.97,
        audio_sample_rate=48000,
    )

    assert probe.model_dump() == {
        "duration_seconds": 120.0,
        "width": 1920,
        "height": 1080,
        "fps": 29.97,
        "audio_sample_rate": 48000,
    }


def test_job_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ClipperJob.model_validate(
            {
                "video_path": "/tmp/input.mp4",
                "output_dir": "/tmp/out",
                "unexpected": True,
            }
        )


def test_job_rejects_non_positive_max_candidates() -> None:
    with pytest.raises(ValidationError):
        ClipperJob.model_validate(
            {
                "video_path": "/tmp/input.mp4",
                "output_dir": "/tmp/out",
                "max_candidates": 0,
            }
        )


def test_output_profile_rejects_unknown_aspect_ratios() -> None:
    with pytest.raises(ValidationError):
        OutputProfile.model_validate({"ratios": ["4:5"]})


def test_render_manifest_rejects_unknown_aspect_ratio_keys() -> None:
    with pytest.raises(ValidationError):
        RenderManifest.model_validate(
            {
                "clip_id": "clip-001",
                "approved": True,
                "outputs": {"4:5": "/tmp/out/clip-001-4x5.mp4"},
            }
        )


def test_media_probe_requires_positive_numeric_fields() -> None:
    with pytest.raises(ValidationError):
        MediaProbe.model_validate(
            {
                "duration_seconds": 0,
                "width": 1920,
                "height": 1080,
                "fps": 29.97,
                "audio_sample_rate": 48000,
            }
        )


def test_candidate_clip_rejects_negative_start_time() -> None:
    with pytest.raises(ValidationError):
        CandidateClip.model_validate(
            {
                "clip_id": "clip-001",
                "start_seconds": -0.1,
                "end_seconds": 10.0,
                "score": 0.5,
                "reasons": ["hook"],
                "spike_categories": ["hook"],
            }
        )


def test_candidate_clip_requires_end_after_start() -> None:
    with pytest.raises(ValidationError):
        CandidateClip.model_validate(
            {
                "clip_id": "clip-001",
                "start_seconds": 10.0,
                "end_seconds": 10.0,
                "score": 0.5,
                "reasons": ["hook"],
                "spike_categories": ["hook"],
            }
        )


def test_candidate_clip_score_must_be_between_zero_and_one() -> None:
    with pytest.raises(ValidationError):
        CandidateClip.model_validate(
            {
                "clip_id": "clip-001",
                "start_seconds": 1.0,
                "end_seconds": 10.0,
                "score": 1.1,
                "reasons": ["hook"],
                "spike_categories": ["hook"],
            }
        )


def test_review_manifest_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ReviewManifest.model_validate(
            {
                "job_id": "job-123",
                "video_path": "/tmp/input.mp4",
                "candidates": [],
                "status": "pending",
            }
        )
