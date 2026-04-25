from __future__ import annotations

import json
from pathlib import Path

import pytest

from clippos.adapters.storage import write_json
from clippos.models.candidate import CandidateClip
from clippos.models.package import (
    PACKAGE_PROMPT_VERSION,
    PackageResponse,
    PublishPack,
)
from clippos.models.review import ReviewManifest
from clippos.models.scoring import ClipBrief, MiningSignals, ScoringRequest
from clippos.pipeline.packaging import (
    PACKAGE_REPORT_FILENAME,
    PACKAGE_REQUEST_FILENAME,
    PACKAGE_RESPONSE_FILENAME,
    PackagingResponseError,
    briefs_for_approved_candidates,
    build_package_brief,
    build_package_prompt,
    build_package_request,
    build_package_schema,
    load_package_request,
    load_package_response,
    resolve_packs,
    write_package_report,
    write_package_request,
    write_pack_artifacts,
)


def _mining_signals() -> MiningSignals:
    return MiningSignals(
        hook=0.8,
        keyword=0.5,
        numeric=0.1,
        interjection=0.1,
        payoff=0.6,
        question_to_answer=0.0,
        motion=0.3,
        shot_change=0.2,
        face_presence=0.7,
        speaker_interaction=0.3,
        delivery_variance=0.4,
        buried_lead=False,
        dangling_question=False,
        rambling_middle=False,
        reasons=["strong hook"],
        spike_categories=["controversy"],
    )


def _brief(clip_id: str, clip_hash: str) -> ClipBrief:
    return ClipBrief(
        clip_id=clip_id,
        clip_hash=clip_hash,
        start_seconds=10.0,
        end_seconds=40.0,
        duration_seconds=30.0,
        transcript=f"transcript for {clip_id}",
        speakers=["speaker_1"],
        mining_score=0.7,
        mining_signals=_mining_signals(),
    )


def _candidate(clip_id: str, *, approved: bool, score: float = 0.8) -> CandidateClip:
    return CandidateClip(
        clip_id=clip_id,
        start_seconds=10.0,
        end_seconds=40.0,
        score=score,
        reasons=["strong hook"],
        spike_categories=["controversy"],
        title=f"Title {clip_id}",
        hook=f"Hook {clip_id}",
        approved=approved,
    )


def _pack(clip_id: str, clip_hash: str) -> dict:
    return {
        "clip_id": clip_id,
        "clip_hash": clip_hash,
        "titles": [f"Angle {i}" for i in range(5)],
        "thumbnail_texts": ["HOT TAKE", "BIG CLAIM", "LAST SHOT"],
        "social_caption": "Caption body that fits within limits.",
        "hashtags": ["#shorts", "#creator", "#podcast", "#viral", "#interview"],
        "hooks": ["First hook line.", "Second hook line."],
    }


def test_briefs_for_approved_candidates_only_includes_approved() -> None:
    scoring_request = ScoringRequest(
        rubric_version="v1",
        job_id="job-1",
        video_path=Path("/tmp/input.mp4"),
        rubric_prompt="prompt",
        response_schema={},
        clips=[_brief("clip-a", "hash-a"), _brief("clip-b", "hash-b")],
    )
    review = ReviewManifest(
        job_id="job-1",
        video_path=Path("/tmp/input.mp4"),
        candidates=[
            _candidate("clip-a", approved=True),
            _candidate("clip-b", approved=False),
        ],
    )

    briefs = briefs_for_approved_candidates(review, scoring_request)

    assert [brief.clip_id for brief in briefs] == ["clip-a"]
    assert briefs[0].clip_hash == "hash-a"
    assert briefs[0].title_hint == "Title clip-a"


def test_briefs_for_approved_candidates_raises_on_missing_brief() -> None:
    scoring_request = ScoringRequest(
        rubric_version="v1",
        job_id="job-1",
        video_path=Path("/tmp/input.mp4"),
        rubric_prompt="prompt",
        response_schema={},
        clips=[_brief("clip-a", "hash-a")],
    )
    review = ReviewManifest(
        job_id="job-1",
        video_path=Path("/tmp/input.mp4"),
        candidates=[_candidate("clip-b", approved=True)],
    )
    with pytest.raises(PackagingResponseError, match="missing clip_id"):
        briefs_for_approved_candidates(review, scoring_request)


def test_build_package_request_embeds_prompt_and_schema() -> None:
    brief = _brief("clip-a", "hash-a")
    request = build_package_request(
        job_id="job-1",
        video_path=Path("/tmp/input.mp4"),
        briefs=[
            build_package_brief(
                candidate=_candidate("clip-a", approved=True), brief=brief
            )
        ],
    )
    assert request.prompt_version == PACKAGE_PROMPT_VERSION
    assert request.package_prompt == build_package_prompt()
    assert request.response_schema == build_package_schema()
    assert request.response_schema["type"] == "object"
    assert len(request.clips) == 1


def test_build_package_request_rejects_empty_briefs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_package_request(job_id="job-1", video_path=Path("/tmp/a.mp4"), briefs=[])


def test_write_and_load_package_request_round_trip(tmp_path: Path) -> None:
    request = build_package_request(
        job_id="job-1",
        video_path=Path("/tmp/input.mp4"),
        briefs=[
            build_package_brief(
                candidate=_candidate("clip-a", approved=True),
                brief=_brief("clip-a", "hash-a"),
            )
        ],
    )
    path = write_package_request(tmp_path, request)
    assert path.name == PACKAGE_REQUEST_FILENAME

    reloaded = load_package_request(tmp_path)
    assert reloaded is not None
    assert reloaded.job_id == request.job_id
    assert reloaded.clips[0].clip_hash == "hash-a"


def test_load_package_response_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(PackagingResponseError, match=PACKAGE_RESPONSE_FILENAME):
        load_package_response(tmp_path)


def test_resolve_packs_enforces_prompt_version_and_job_id(tmp_path: Path) -> None:
    request = build_package_request(
        job_id="job-1",
        video_path=Path("/tmp/input.mp4"),
        briefs=[
            build_package_brief(
                candidate=_candidate("clip-a", approved=True),
                brief=_brief("clip-a", "hash-a"),
            )
        ],
    )

    mismatched_version = PackageResponse(
        prompt_version="other",
        job_id="job-1",
        packs=[PublishPack.model_validate(_pack("clip-a", "hash-a"))],
    )
    with pytest.raises(PackagingResponseError, match="prompt_version"):
        resolve_packs(request, mismatched_version)

    mismatched_job = PackageResponse(
        prompt_version=request.prompt_version,
        job_id="other-job",
        packs=[PublishPack.model_validate(_pack("clip-a", "hash-a"))],
    )
    with pytest.raises(PackagingResponseError, match="job_id"):
        resolve_packs(request, mismatched_job)


def test_resolve_packs_detects_missing_or_swapped_clip() -> None:
    request = build_package_request(
        job_id="job-1",
        video_path=Path("/tmp/input.mp4"),
        briefs=[
            build_package_brief(
                candidate=_candidate("clip-a", approved=True),
                brief=_brief("clip-a", "hash-a"),
            )
        ],
    )

    missing = PackageResponse(
        prompt_version=request.prompt_version,
        job_id=request.job_id,
        packs=[PublishPack.model_validate(_pack("clip-b", "hash-b"))],
    )
    with pytest.raises(PackagingResponseError, match="missing pack"):
        resolve_packs(request, missing)

    swapped_pack = _pack("clip-b", "hash-a")  # wrong clip_id for hash-a
    swapped = PackageResponse(
        prompt_version=request.prompt_version,
        job_id=request.job_id,
        packs=[PublishPack.model_validate(swapped_pack)],
    )
    with pytest.raises(PackagingResponseError, match="returned clip_id"):
        resolve_packs(request, swapped)


def test_write_pack_artifacts_and_report_fan_out_per_clip(tmp_path: Path) -> None:
    pack_a = PublishPack.model_validate(_pack("clip-a", "hash-a"))
    pack_b = PublishPack.model_validate(_pack("clip-b", "hash-b"))

    def resolver(clip_id: str) -> Path:
        return tmp_path / "renders" / clip_id

    pack_paths = write_pack_artifacts(
        workspace_dir=tmp_path,
        clip_dir_for=resolver,
        packs=[pack_a, pack_b],
    )

    assert pack_paths["clip-a"] == tmp_path / "renders" / "clip-a" / "package.json"
    assert pack_paths["clip-b"].exists()
    loaded = json.loads(pack_paths["clip-a"].read_text(encoding="utf-8"))
    assert loaded["clip_id"] == "clip-a"

    report_path = write_package_report(
        workspace_dir=tmp_path,
        job_id="job-1",
        video_path=Path("/tmp/input.mp4"),
        pack_paths=pack_paths,
    )
    assert report_path.name == PACKAGE_REPORT_FILENAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert {entry["clip_id"] for entry in report["packs"]} == {"clip-a", "clip-b"}
    for entry in report["packs"]:
        assert not entry["pack_path"].startswith("/")  # workspace-relative


def test_load_package_response_validates_bad_json(tmp_path: Path) -> None:
    (tmp_path / PACKAGE_RESPONSE_FILENAME).write_text("not-json", encoding="utf-8")
    with pytest.raises(PackagingResponseError, match="not valid JSON"):
        load_package_response(tmp_path)


def test_load_package_response_validates_schema(tmp_path: Path) -> None:
    write_json(
        tmp_path / PACKAGE_RESPONSE_FILENAME,
        {"prompt_version": PACKAGE_PROMPT_VERSION, "job_id": "job-1"},
    )
    with pytest.raises(PackagingResponseError, match="packaging contract"):
        load_package_response(tmp_path)
