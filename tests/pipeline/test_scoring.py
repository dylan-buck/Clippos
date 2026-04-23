from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipper.adapters.rubric import RUBRIC_VERSION, build_rubric_prompt
from clipper.models.candidate import CandidateClip
from clipper.models.scoring import (
    ClipBrief,
    ClipScore,
    MiningSignals,
    RubricScores,
    ScoringRequest,
    ScoringResponse,
)
from clipper.pipeline.candidates import ScoredWindow, WindowSignals
from clipper.pipeline.scoring import (
    SCORING_REQUEST_FILENAME,
    SCORING_RESPONSE_FILENAME,
    ScoringResponseError,
    build_clip_brief,
    build_scoring_request,
    compute_clip_hash,
    load_cached_score,
    load_scoring_request,
    load_scoring_response,
    persist_cached_score,
    resolve_scores,
    scores_to_model_payload,
    scoring_cache_dir,
    write_scoring_request,
)
from clipper.pipeline.transcribe import TranscriptSegment


def _segment(**overrides) -> TranscriptSegment:
    defaults = {
        "speaker": "speaker_1",
        "start_seconds": 0.0,
        "end_seconds": 8.0,
        "text": "Here's the thing about secrets.",
        "words": [],
    }
    defaults.update(overrides)
    return TranscriptSegment(**defaults)


def _signals(**overrides) -> WindowSignals:
    defaults = {
        "hook": 0.8,
        "keyword": 0.6,
        "numeric": 0.1,
        "interjection": 0.05,
        "payoff": 0.5,
        "question_to_answer": 0.0,
        "motion": 0.4,
        "shot_change": 0.2,
        "face_presence": 0.7,
        "speaker_interaction": 0.0,
        "delivery_variance": 0.2,
        "buried_lead": False,
        "dangling_question": False,
        "rambling_middle": False,
    }
    defaults.update(overrides)
    return WindowSignals(**defaults)


def _scored_window(
    segments: tuple[TranscriptSegment, ...], *, score: float = 0.75
) -> ScoredWindow:
    return ScoredWindow(segments=segments, signals=_signals(), score=score)


def _candidate(**overrides) -> CandidateClip:
    defaults = {
        "clip_id": "clip-000",
        "start_seconds": 0.0,
        "end_seconds": 18.0,
        "score": 0.75,
        "reasons": ["strong hook"],
        "spike_categories": ["controversy"],
    }
    defaults.update(overrides)
    return CandidateClip(**defaults)


def _rubric_scores(**overrides) -> RubricScores:
    defaults = {
        "hook": 0.85,
        "shareability": 0.7,
        "standalone_clarity": 0.8,
        "payoff": 0.75,
        "delivery_energy": 0.65,
        "quotability": 0.55,
    }
    defaults.update(overrides)
    return RubricScores(**defaults)


def _clip_score(**overrides) -> ClipScore:
    defaults = {
        "clip_id": "clip-000",
        "clip_hash": "deadbeefdeadbeef",
        "rubric": _rubric_scores(),
        "spike_categories": ["taboo"],
        "penalties": [],
        "final_score": 0.82,
        "title": "Secret reveal",
        "hook": "Nobody tells you this",
        "reasons": ["strong hook", "payoff"],
    }
    defaults.update(overrides)
    return ClipScore(**defaults)


def test_compute_clip_hash_is_stable_for_identical_input() -> None:
    a = compute_clip_hash(
        transcript="hello world",
        start_seconds=1.0,
        end_seconds=5.0,
        rubric_version="1.0.0",
    )
    b = compute_clip_hash(
        transcript="hello world",
        start_seconds=1.0,
        end_seconds=5.0,
        rubric_version="1.0.0",
    )
    assert a == b
    assert len(a) == 16


def test_compute_clip_hash_differs_when_transcript_changes() -> None:
    base = compute_clip_hash(
        transcript="hello",
        start_seconds=0.0,
        end_seconds=1.0,
        rubric_version="1.0.0",
    )
    changed = compute_clip_hash(
        transcript="hello world",
        start_seconds=0.0,
        end_seconds=1.0,
        rubric_version="1.0.0",
    )
    assert base != changed


def test_compute_clip_hash_differs_when_rubric_version_changes() -> None:
    base = compute_clip_hash(
        transcript="hello",
        start_seconds=0.0,
        end_seconds=1.0,
        rubric_version="1.0.0",
    )
    v2 = compute_clip_hash(
        transcript="hello",
        start_seconds=0.0,
        end_seconds=1.0,
        rubric_version="2.0.0",
    )
    assert base != v2


def test_compute_clip_hash_differs_when_time_bounds_shift() -> None:
    base = compute_clip_hash(
        transcript="hello",
        start_seconds=0.0,
        end_seconds=1.0,
        rubric_version="1.0.0",
    )
    shifted = compute_clip_hash(
        transcript="hello",
        start_seconds=0.1,
        end_seconds=1.0,
        rubric_version="1.0.0",
    )
    assert base != shifted


def test_build_clip_brief_includes_transcript_speakers_and_signals() -> None:
    segments = (
        _segment(speaker="speaker_1", text="Here's the wild part."),
        _segment(
            speaker="speaker_2",
            start_seconds=8.0,
            end_seconds=18.0,
            text="Turns out it worked.",
        ),
    )
    window = _scored_window(segments, score=0.75)
    candidate = _candidate(end_seconds=18.0, score=0.75)

    brief = build_clip_brief(candidate=candidate, scored=window)

    assert brief.clip_id == "clip-000"
    assert brief.transcript == "Here's the wild part. Turns out it worked."
    assert brief.speakers == ["speaker_1", "speaker_2"]
    assert brief.duration_seconds == pytest.approx(18.0)
    assert brief.mining_score == pytest.approx(0.75)
    assert brief.mining_signals.reasons == ["strong hook"]
    assert brief.mining_signals.spike_categories == ["controversy"]


def test_build_clip_brief_derives_hash_from_rubric_version_and_window() -> None:
    segments = (_segment(text="identical transcript"),)
    window = _scored_window(segments)
    candidate = _candidate(start_seconds=0.0, end_seconds=8.0)
    brief = build_clip_brief(candidate=candidate, scored=window)
    expected = compute_clip_hash(
        transcript="identical transcript",
        start_seconds=0.0,
        end_seconds=8.0,
        rubric_version=RUBRIC_VERSION,
    )
    assert brief.clip_hash == expected


def test_build_scoring_request_embeds_rubric_prompt_and_schema() -> None:
    segments = (_segment(text="wild story"),)
    window = _scored_window(segments)
    candidate = _candidate(start_seconds=0.0, end_seconds=8.0)
    brief = build_clip_brief(candidate=candidate, scored=window)

    request = build_scoring_request(
        job_id="job-xyz",
        video_path=Path("/tmp/input.mp4"),
        briefs=[brief],
    )
    assert request.rubric_version == RUBRIC_VERSION
    assert request.rubric_prompt == build_rubric_prompt()
    assert (
        request.response_schema["properties"]["rubric_version"]["const"]
        == RUBRIC_VERSION
    )
    assert request.clips == [brief]


def test_write_scoring_request_and_load_roundtrips(tmp_path: Path) -> None:
    segments = (_segment(text="hello"),)
    window = _scored_window(segments)
    brief = build_clip_brief(candidate=_candidate(end_seconds=8.0), scored=window)
    request = build_scoring_request(
        job_id="job-1",
        video_path=tmp_path / "video.mp4",
        briefs=[brief],
    )

    path = write_scoring_request(tmp_path, request)
    assert path == tmp_path / SCORING_REQUEST_FILENAME

    loaded = load_scoring_request(tmp_path)
    assert loaded == request


def test_load_scoring_request_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_scoring_request(tmp_path) is None


def test_load_scoring_response_raises_on_malformed_json(tmp_path: Path) -> None:
    (tmp_path / SCORING_RESPONSE_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(ScoringResponseError):
        load_scoring_response(tmp_path)


def test_load_scoring_response_raises_on_contract_violation(tmp_path: Path) -> None:
    (tmp_path / SCORING_RESPONSE_FILENAME).write_text(
        json.dumps(
            {"rubric_version": RUBRIC_VERSION, "job_id": "x", "scores": "not a list"}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ScoringResponseError):
        load_scoring_response(tmp_path)


def test_load_scoring_response_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_scoring_response(tmp_path) is None


def test_cache_roundtrip_reads_persisted_score(tmp_path: Path) -> None:
    score = _clip_score(clip_hash="aaaabbbbccccdddd")
    persist_cached_score(tmp_path, score)
    cache_path = scoring_cache_dir(tmp_path) / "aaaabbbbccccdddd.json"
    assert cache_path.exists()
    loaded = load_cached_score(tmp_path, "aaaabbbbccccdddd")
    assert loaded == score


def test_load_cached_score_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_cached_score(tmp_path, "missing") is None


def test_resolve_scores_returns_none_when_no_request_present(tmp_path: Path) -> None:
    assert resolve_scores(tmp_path) is None


def test_resolve_scores_returns_none_when_response_is_missing(tmp_path: Path) -> None:
    segments = (_segment(text="hello"),)
    brief = build_clip_brief(
        candidate=_candidate(end_seconds=8.0),
        scored=_scored_window(segments),
    )
    request = build_scoring_request(
        job_id="job-1",
        video_path=tmp_path / "v.mp4",
        briefs=[brief],
    )
    write_scoring_request(tmp_path, request)
    assert resolve_scores(tmp_path) is None


def test_resolve_scores_uses_response_and_persists_to_cache(tmp_path: Path) -> None:
    segments = (_segment(text="wild"),)
    brief = build_clip_brief(
        candidate=_candidate(end_seconds=8.0),
        scored=_scored_window(segments),
    )
    request = build_scoring_request(
        job_id="job-1",
        video_path=tmp_path / "v.mp4",
        briefs=[brief],
    )
    write_scoring_request(tmp_path, request)
    score = _clip_score(clip_id="clip-000", clip_hash=brief.clip_hash)
    ScoringResponse(
        rubric_version=RUBRIC_VERSION,
        job_id="job-1",
        scores=[score],
    ).model_dump(mode="json")
    (tmp_path / SCORING_RESPONSE_FILENAME).write_text(
        json.dumps(
            {
                "rubric_version": RUBRIC_VERSION,
                "job_id": "job-1",
                "scores": [score.model_dump(mode="json")],
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_scores(tmp_path)
    assert resolved is not None
    assert len(resolved) == 1
    assert resolved[0].clip_hash == brief.clip_hash
    cached = load_cached_score(tmp_path, brief.clip_hash)
    assert cached == score


def test_resolve_scores_uses_cache_when_response_absent(tmp_path: Path) -> None:
    segments = (_segment(text="cached"),)
    brief = build_clip_brief(
        candidate=_candidate(end_seconds=8.0),
        scored=_scored_window(segments),
    )
    request = build_scoring_request(
        job_id="job-1",
        video_path=tmp_path / "v.mp4",
        briefs=[brief],
    )
    write_scoring_request(tmp_path, request)
    cached = _clip_score(
        clip_id="clip-007", clip_hash=brief.clip_hash, title="From cache"
    )
    persist_cached_score(tmp_path, cached)

    resolved = resolve_scores(tmp_path)
    assert resolved is not None
    assert len(resolved) == 1
    assert resolved[0].clip_id == "clip-000"  # rehomed to current rank
    assert resolved[0].title == "From cache"


def test_resolve_scores_raises_on_rubric_version_mismatch(tmp_path: Path) -> None:
    segments = (_segment(text="a"),)
    brief = build_clip_brief(
        candidate=_candidate(end_seconds=8.0),
        scored=_scored_window(segments),
    )
    request = build_scoring_request(
        job_id="job-1",
        video_path=tmp_path / "v.mp4",
        briefs=[brief],
    )
    write_scoring_request(tmp_path, request)
    (tmp_path / SCORING_RESPONSE_FILENAME).write_text(
        json.dumps(
            {
                "rubric_version": "9.9.9",
                "job_id": "job-1",
                "scores": [
                    _clip_score(clip_hash=brief.clip_hash).model_dump(mode="json")
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ScoringResponseError):
        resolve_scores(tmp_path)


def test_resolve_scores_raises_on_job_id_mismatch(tmp_path: Path) -> None:
    segments = (_segment(text="b"),)
    brief = build_clip_brief(
        candidate=_candidate(end_seconds=8.0),
        scored=_scored_window(segments),
    )
    request = build_scoring_request(
        job_id="job-1",
        video_path=tmp_path / "v.mp4",
        briefs=[brief],
    )
    write_scoring_request(tmp_path, request)
    (tmp_path / SCORING_RESPONSE_FILENAME).write_text(
        json.dumps(
            {
                "rubric_version": RUBRIC_VERSION,
                "job_id": "job-MISMATCH",
                "scores": [
                    _clip_score(clip_hash=brief.clip_hash).model_dump(mode="json")
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ScoringResponseError):
        resolve_scores(tmp_path)


def test_resolve_scores_raises_on_clip_id_mismatch(tmp_path: Path) -> None:
    segments = (_segment(text="c"),)
    brief = build_clip_brief(
        candidate=_candidate(end_seconds=8.0),
        scored=_scored_window(segments),
    )
    request = build_scoring_request(
        job_id="job-1",
        video_path=tmp_path / "v.mp4",
        briefs=[brief],
    )
    write_scoring_request(tmp_path, request)
    bad = _clip_score(clip_id="clip-999", clip_hash=brief.clip_hash)
    (tmp_path / SCORING_RESPONSE_FILENAME).write_text(
        json.dumps(
            {
                "rubric_version": RUBRIC_VERSION,
                "job_id": "job-1",
                "scores": [bad.model_dump(mode="json")],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ScoringResponseError):
        resolve_scores(tmp_path)


def test_resolve_scores_returns_none_when_any_clip_is_unscored(tmp_path: Path) -> None:
    first_segments = (_segment(text="first"),)
    second_segments = (_segment(start_seconds=10.0, end_seconds=18.0, text="second"),)
    first_brief = build_clip_brief(
        candidate=_candidate(clip_id="clip-000", end_seconds=8.0),
        scored=_scored_window(first_segments),
    )
    second_brief = build_clip_brief(
        candidate=_candidate(clip_id="clip-001", start_seconds=10.0, end_seconds=18.0),
        scored=_scored_window(second_segments),
    )
    request = build_scoring_request(
        job_id="job-1",
        video_path=tmp_path / "v.mp4",
        briefs=[first_brief, second_brief],
    )
    write_scoring_request(tmp_path, request)
    # only score the first clip
    (tmp_path / SCORING_RESPONSE_FILENAME).write_text(
        json.dumps(
            {
                "rubric_version": RUBRIC_VERSION,
                "job_id": "job-1",
                "scores": [
                    _clip_score(
                        clip_id="clip-000", clip_hash=first_brief.clip_hash
                    ).model_dump(mode="json")
                ],
            }
        ),
        encoding="utf-8",
    )

    assert resolve_scores(tmp_path) is None


def test_scores_to_model_payload_shape_is_build_review_manifest_friendly() -> None:
    payload = scores_to_model_payload([_clip_score()])
    assert payload == [
        {
            "clip_id": "clip-000",
            "title": "Secret reveal",
            "hook": "Nobody tells you this",
            "reasons": ["strong hook", "payoff"],
            "final_score": 0.82,
        }
    ]


# Typing convenience to appease the test-only stubs without importing at runtime.
__all__ = [
    "ClipBrief",
    "MiningSignals",
    "ScoringRequest",
]
