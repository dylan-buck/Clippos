"""Tests for the v1.1 video-brief pipeline stage.

Covers:
- transcript excerpt construction (head/tail truncation when long)
- brief request building (Pydantic contract, prompt + schema embedded)
- brief response loading (validation errors surface cleanly)
- brief caching across reruns (rubric_version + job_id matched)
- BriefResponseError surfacing (rubric mismatch, job mismatch)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from clippos.adapters.brief import (
    BRIEF_VERSION,
    build_brief_prompt,
    build_brief_response_schema,
)
from clippos.pipeline.brief import (
    BRIEF_CACHE_FILENAME,
    BRIEF_REQUEST_FILENAME,
    MAX_TRANSCRIPT_CHARS,
    BriefResponseError,
    brief_cache_path,
    brief_request_path,
    brief_response_path,
    build_brief_request,
    build_transcript_excerpt,
    load_brief_request,
    persist_cached_brief,
    resolve_brief,
    write_brief_request,
)
from clippos.models.scoring import (
    VideoBrief,
    VideoBriefRequest,
    VideoBriefResponse,
)
from clippos.pipeline.transcribe import build_transcript_timeline

VALID_EXPECTED_PATTERNS = [
    "the warm-welcome moment",
    "a specific guest insight",
    "a clear payoff after setup",
]


def _timeline_with_text(text_per_segment: list[tuple[str, str, float, float]]):
    """Build a TranscriptTimeline from (speaker, text, start, end) tuples."""
    return build_transcript_timeline(
        {
            "segments": [
                {
                    "speaker": speaker,
                    "start_seconds": start,
                    "end_seconds": end,
                    "text": text,
                    "words": [],
                }
                for speaker, text, start, end in text_per_segment
            ]
        }
    )


# ---------- transcript excerpt building ----------


def test_build_transcript_excerpt_returns_full_text_when_short() -> None:
    timeline = _timeline_with_text(
        [
            ("host", "Welcome to the show.", 0.0, 5.0),
            ("guest", "Thanks for having me.", 5.0, 10.0),
        ]
    )
    excerpt, truncated = build_transcript_excerpt(timeline)

    assert truncated is False
    assert "Welcome to the show." in excerpt
    assert "Thanks for having me." in excerpt
    # Speaker + timestamp formatting is preserved so the model can see
    # who said what when.
    assert "[host" in excerpt
    assert "[guest" in excerpt


def test_build_transcript_excerpt_truncates_long_transcripts_with_marker() -> None:
    """Head + tail are preserved; middle is dropped with a clear marker
    so the model knows context is missing. Both ends matter for spine
    detection (lede + wrap-up are where editors find theme content)."""
    long_text = "x " * 25_000  # ~50k chars per segment, way over the cap
    timeline = _timeline_with_text(
        [
            ("host", "INTRO TOKEN " + long_text, 0.0, 60.0),
            ("host", long_text + " OUTRO TOKEN", 60.0, 120.0),
        ]
    )

    excerpt, truncated = build_transcript_excerpt(timeline)

    assert truncated is True
    assert "INTRO TOKEN" in excerpt, "head must be preserved"
    assert "OUTRO TOKEN" in excerpt, "tail must be preserved"
    assert "transcript truncated for brief" in excerpt
    # Bounded length so the model token cost stays predictable.
    assert len(excerpt) <= MAX_TRANSCRIPT_CHARS + 500  # marker has small overhead


# ---------- brief request building ----------


def test_build_brief_request_carries_prompt_schema_and_speakers() -> None:
    timeline = _timeline_with_text(
        [
            ("host", "Welcome.", 0.0, 5.0),
            ("guest", "Hi.", 5.0, 10.0),
            ("host", "Let's dig in.", 10.0, 30.0),
        ]
    )
    request = build_brief_request(
        job_id="job-abc",
        video_path=Path("/tmp/input.mp4"),
        transcript_timeline=timeline,
    )

    assert isinstance(request, VideoBriefRequest)
    assert request.rubric_version == BRIEF_VERSION
    assert request.job_id == "job-abc"
    assert request.transcript_truncated is False
    assert request.duration_seconds == 30.0
    # Speaker order preserved by first-appearance.
    assert request.speakers == ["host", "guest"]
    # Prompt + schema embedded so the harness has everything in one
    # request file.
    assert request.brief_prompt == build_brief_prompt()
    assert request.response_schema == build_brief_response_schema()


def test_write_and_load_brief_request_roundtrips(tmp_path: Path) -> None:
    timeline = _timeline_with_text(
        [
            ("host", "Welcome.", 0.0, 5.0),
            ("guest", "Hi.", 5.0, 10.0),
        ]
    )
    request = build_brief_request(
        job_id="job-rt", video_path=Path("/tmp/v.mp4"), transcript_timeline=timeline
    )

    written_path = write_brief_request(tmp_path, request)
    assert written_path == brief_request_path(tmp_path)
    assert written_path.name == BRIEF_REQUEST_FILENAME

    loaded = load_brief_request(tmp_path)
    assert loaded is not None
    assert loaded == request


def test_load_brief_request_returns_none_on_missing_file(tmp_path: Path) -> None:
    assert load_brief_request(tmp_path) is None


def test_load_brief_request_returns_none_on_invalid_json(tmp_path: Path) -> None:
    brief_request_path(tmp_path).write_text("not valid json", encoding="utf-8")
    assert load_brief_request(tmp_path) is None


# ---------- brief response loading + integrity ----------


def _seed_request_response(
    workspace: Path,
    *,
    job_id: str = "job-1",
    response_job_id: str | None = None,
    response_rubric_version: str | None = None,
    brief_job_id: str | None = None,
    brief_rubric_version: str | None = None,
) -> VideoBrief:
    timeline = _timeline_with_text(
        [("host", "Welcome.", 0.0, 5.0), ("guest", "Hi.", 5.0, 10.0)]
    )
    request = build_brief_request(
        job_id=job_id,
        video_path=Path("/tmp/v.mp4"),
        transcript_timeline=timeline,
    )
    write_brief_request(workspace, request)

    brief = VideoBrief(
        rubric_version=brief_rubric_version or BRIEF_VERSION,
        job_id=brief_job_id or job_id,
        theme="A two-speaker conversation about welcoming guests.",
        video_format="podcast interview, two speakers",
        expected_viral_patterns=VALID_EXPECTED_PATTERNS,
        anti_patterns=["intro chitchat without payoff"],
    )
    response = VideoBriefResponse(
        rubric_version=response_rubric_version or BRIEF_VERSION,
        job_id=response_job_id or job_id,
        brief=brief,
    )
    brief_response_path(workspace).write_text(
        json.dumps(response.model_dump(mode="json")),
        encoding="utf-8",
    )
    return brief


def test_resolve_brief_returns_brief_and_caches_when_response_valid(
    tmp_path: Path,
) -> None:
    expected_brief = _seed_request_response(tmp_path)
    resolved = resolve_brief(tmp_path)

    assert resolved == expected_brief
    # Cache must be populated so a follow-up advance call without a fresh
    # response file still returns the brief (mirrors scoring per-clip cache).
    assert (tmp_path / BRIEF_CACHE_FILENAME).exists()


def test_resolve_brief_uses_cache_when_response_absent(tmp_path: Path) -> None:
    expected_brief = _seed_request_response(tmp_path)
    # Run once to populate cache, then remove the response file and verify
    # cached brief still resolves.
    resolve_brief(tmp_path)
    brief_response_path(tmp_path).unlink()

    cached_resolution = resolve_brief(tmp_path)
    assert cached_resolution == expected_brief


def test_resolve_brief_returns_none_when_no_response_or_cache(
    tmp_path: Path,
) -> None:
    timeline = _timeline_with_text([("host", "Hi.", 0.0, 5.0)])
    request = build_brief_request(
        job_id="job-empty", video_path=Path("/tmp/v.mp4"),
        transcript_timeline=timeline,
    )
    write_brief_request(tmp_path, request)
    # No response, no cache.
    assert resolve_brief(tmp_path) is None


def test_resolve_brief_returns_none_when_request_missing(tmp_path: Path) -> None:
    # No request file means we have no contract to validate against —
    # return None so the caller drops the brief gracefully.
    assert resolve_brief(tmp_path) is None


def test_resolve_brief_raises_on_rubric_version_mismatch(tmp_path: Path) -> None:
    _seed_request_response(
        tmp_path,
        response_rubric_version="999.999.999",
        brief_rubric_version="999.999.999",
    )
    with pytest.raises(BriefResponseError, match="rubric_version"):
        resolve_brief(tmp_path)


def test_resolve_brief_raises_on_job_id_mismatch(tmp_path: Path) -> None:
    _seed_request_response(tmp_path, response_job_id="other-job")
    with pytest.raises(BriefResponseError, match="job_id"):
        resolve_brief(tmp_path)


def test_resolve_brief_raises_on_invalid_response_json(tmp_path: Path) -> None:
    timeline = _timeline_with_text([("host", "Hi.", 0.0, 5.0)])
    request = build_brief_request(
        job_id="job-bad", video_path=Path("/tmp/v.mp4"),
        transcript_timeline=timeline,
    )
    write_brief_request(tmp_path, request)
    brief_response_path(tmp_path).write_text("not valid json", encoding="utf-8")

    with pytest.raises(BriefResponseError, match="not valid JSON"):
        resolve_brief(tmp_path)


def test_resolve_brief_invalidates_cache_on_rubric_change(tmp_path: Path) -> None:
    """A cached brief from a prior rubric version must not satisfy a
    request under the current version. Otherwise an upgrade would
    silently keep using stale frames."""
    _seed_request_response(tmp_path)
    resolve_brief(tmp_path)
    brief_response_path(tmp_path).unlink()

    # Manually corrupt the cache to look like an old-version cache.
    cache = json.loads(brief_cache_path(tmp_path).read_text(encoding="utf-8"))
    cache["rubric_version"] = "999.999.999"
    brief_cache_path(tmp_path).write_text(json.dumps(cache), encoding="utf-8")

    # Now resolve_brief should treat the cache as invalid (not raise; just
    # return None so the caller re-emits the handoff).
    assert resolve_brief(tmp_path) is None


# ---------- VideoBrief Pydantic contract ----------


def test_video_brief_rejects_empty_theme() -> None:
    with pytest.raises(ValidationError):
        VideoBrief(
            rubric_version=BRIEF_VERSION,
            job_id="job-x",
            theme="",  # min_length=1
            video_format="podcast",
            expected_viral_patterns=[],
        )


def test_video_brief_enforces_pattern_counts() -> None:
    with pytest.raises(ValidationError):
        VideoBrief(
            rubric_version=BRIEF_VERSION,
            job_id="job-x",
            theme="A theme.",
            video_format="podcast",
            expected_viral_patterns=["too few"],
        )

    with pytest.raises(ValidationError):
        VideoBrief(
            rubric_version=BRIEF_VERSION,
            job_id="job-x",
            theme="A theme.",
            video_format="podcast",
            expected_viral_patterns=[
                "one",
                "two",
                "three",
                "four",
                "five",
                "too many",
            ],
        )

    with pytest.raises(ValidationError):
        VideoBrief(
            rubric_version=BRIEF_VERSION,
            job_id="job-x",
            theme="A theme.",
            video_format="podcast",
            expected_viral_patterns=VALID_EXPECTED_PATTERNS,
            anti_patterns=["one", "two", "three", "too many"],
        )


def test_video_brief_accepts_optional_fields() -> None:
    brief = VideoBrief(
        rubric_version=BRIEF_VERSION,
        job_id="job-x",
        theme="A theme.",
        video_format="podcast",
        expected_viral_patterns=VALID_EXPECTED_PATTERNS,
        # anti_patterns has default; audience/tone/notes are optional.
    )
    assert brief.audience is None
    assert brief.tone is None
    assert brief.notes is None
    assert brief.anti_patterns == []


def test_brief_response_schema_enforces_pattern_counts() -> None:
    schema = build_brief_response_schema()
    brief_props = schema["properties"]["brief"]["properties"]

    assert brief_props["expected_viral_patterns"]["minItems"] == 3
    assert brief_props["expected_viral_patterns"]["maxItems"] == 5
    assert brief_props["anti_patterns"]["maxItems"] == 3


def test_persist_cached_brief_writes_canonical_path(tmp_path: Path) -> None:
    brief = VideoBrief(
        rubric_version=BRIEF_VERSION,
        job_id="job-cache",
        theme="A cached brief.",
        video_format="podcast",
        expected_viral_patterns=VALID_EXPECTED_PATTERNS,
    )
    path = persist_cached_brief(tmp_path, brief)
    assert path.name == BRIEF_CACHE_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["job_id"] == "job-cache"
    assert payload["theme"] == "A cached brief."
