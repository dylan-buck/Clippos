from pathlib import Path

import pytest

from clippos.models.candidate import CandidateClip
from clippos.pipeline.review import build_review_manifest


@pytest.fixture
def sample_candidates() -> list[CandidateClip]:
    return [
        CandidateClip(
            clip_id="clip-1",
            start_seconds=10.0,
            end_seconds=25.0,
            score=0.94,
            reasons=["shareability"],
            spike_categories=["absurdity"],
            title="Original candidate title",
            hook="Original candidate hook",
        )
    ]


def test_build_review_manifest_enriches_candidates_with_titles(
    sample_candidates, tmp_path: Path
) -> None:
    manifest = build_review_manifest(
        job_id="job-123",
        video_path=tmp_path / "input.mp4",
        candidates=sample_candidates,
        model_scores=[
            {
                "clip_id": "clip-1",
                "title": "He admits the hidden tradeoff",
                "hook": "Nobody tells you this part",
                "reasons": ["strong hook", "clear payoff"],
            }
        ],
    )

    assert manifest.candidates[0].title == "He admits the hidden tradeoff"
    assert manifest.candidates[0].hook == "Nobody tells you this part"
    assert manifest.candidates[0].reasons == ["strong hook", "clear payoff"]


def test_build_review_manifest_preserves_existing_metadata_when_model_fields_missing(
    sample_candidates, tmp_path: Path
) -> None:
    manifest = build_review_manifest(
        job_id="job-123",
        video_path=tmp_path / "input.mp4",
        candidates=sample_candidates,
        model_scores=[
            {
                "clip_id": "clip-1",
                "reasons": ["model kept the original rationale focused"],
            }
        ],
    )

    assert manifest.candidates[0].title == "Original candidate title"
    assert manifest.candidates[0].hook == "Original candidate hook"
    assert manifest.candidates[0].reasons == [
        "model kept the original rationale focused"
    ]


def test_build_review_manifest_preserves_existing_metadata_when_model_fields_are_none(
    sample_candidates, tmp_path: Path
) -> None:
    manifest = build_review_manifest(
        job_id="job-123",
        video_path=tmp_path / "input.mp4",
        candidates=sample_candidates,
        model_scores=[
            {
                "clip_id": "clip-1",
                "title": None,
                "hook": None,
                "reasons": None,
            }
        ],
    )

    assert manifest.candidates[0].title == "Original candidate title"
    assert manifest.candidates[0].hook == "Original candidate hook"
    assert manifest.candidates[0].reasons == ["shareability"]


def test_build_review_manifest_ignores_malformed_score_rows_without_clip_id(
    sample_candidates, tmp_path: Path
) -> None:
    manifest = build_review_manifest(
        job_id="job-123",
        video_path=tmp_path / "input.mp4",
        candidates=sample_candidates,
        model_scores=[
            {
                "title": "Malformed row should be ignored",
                "hook": "Missing clip id",
                "reasons": ["invalid"],
            },
            {
                "clip_id": "clip-1",
                "title": "Valid row still applies",
                "hook": "The matching score row survives",
                "reasons": ["valid enrichment"],
            },
        ],
    )

    assert manifest.candidates[0].title == "Valid row still applies"
    assert manifest.candidates[0].hook == "The matching score row survives"
    assert manifest.candidates[0].reasons == ["valid enrichment"]
