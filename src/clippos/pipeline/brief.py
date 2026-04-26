"""Brief stage: pre-mine model handoff that produces a VideoBrief.

The brief is one model handoff per video. The orchestrator runs it
between mine and score so the per-clip scoring stage receives a global
frame about what makes THIS video clippable. See docs/v1.1.md for
motivation and the dogfood case study.

Wire shape mirrors the scoring handoff:

    brief-request.json  (engine writes; harness reads)
    brief-response.json (harness writes; engine reads)
    brief-cache.json    (engine persists last good response; survives
                         reruns when the harness has not yet authored
                         the response on this turn)

Long transcripts are sampled to ~30k chars before being embedded in the
request — the brief needs representative moments across the full timeline,
not every word, and keeping requests small bounds the model token cost.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from clippos.adapters.brief import (
    BRIEF_VERSION,
    build_brief_prompt,
    build_brief_response_schema,
)
from clippos.adapters.storage import read_json, write_json
from clippos.models.scoring import (
    VideoBrief,
    VideoBriefRequest,
    VideoBriefResponse,
)
from clippos.pipeline.transcribe import TranscriptTimeline

BRIEF_REQUEST_FILENAME = "brief-request.json"
BRIEF_RESPONSE_FILENAME = "brief-response.json"
BRIEF_CACHE_FILENAME = "brief-cache.json"

# Max characters of transcript to embed in the brief request. The brief
# only needs the global shape, so we don't need every word; keeping the
# request small bounds model token cost. 30k chars is roughly 7-8k tokens
# for English transcript, which fits comfortably alongside the prompt
# even on smaller-context harness models.
MAX_TRANSCRIPT_CHARS = 30_000

# How many characters to keep from the head and tail when truncating. The
# remaining budget is spent on deterministic middle samples spread across the
# timeline so the brief author does not overfit to the intro/outro.
TRUNCATION_HEAD_CHARS = 18_000
TRUNCATION_TAIL_CHARS = 10_000
MIDDLE_SAMPLE_COUNT = 5


class BriefResponseError(RuntimeError):
    """Raised when brief-response.json is unreadable or invalid."""


def brief_request_path(workspace_dir: Path) -> Path:
    return workspace_dir / BRIEF_REQUEST_FILENAME


def brief_response_path(workspace_dir: Path) -> Path:
    return workspace_dir / BRIEF_RESPONSE_FILENAME


def brief_cache_path(workspace_dir: Path) -> Path:
    return workspace_dir / BRIEF_CACHE_FILENAME


def build_transcript_excerpt(
    timeline: TranscriptTimeline,
    *,
    max_chars: int = MAX_TRANSCRIPT_CHARS,
    head_chars: int = TRUNCATION_HEAD_CHARS,
    tail_chars: int = TRUNCATION_TAIL_CHARS,
) -> tuple[str, bool]:
    """Render the transcript as a single representative string.

    Returns ``(excerpt, truncated)``. When truncating, keep the beginning and
    ending while also sampling deterministic middle segments across the full
    timeline. This avoids a brief that captures the intro theme but misses a
    different recurring format later in the video.
    """
    parts: list[str] = []
    for segment in timeline.segments:
        speaker = segment.speaker or "SPEAKER"
        parts.append(f"[{speaker} {segment.start_seconds:.1f}s] {segment.text.strip()}")
    full = "\n".join(parts)
    if len(full) <= max_chars:
        return full, False

    head = full[:head_chars].rstrip()
    tail = full[-tail_chars:].lstrip() if tail_chars > 0 else ""
    middle_budget = max(max_chars - head_chars - tail_chars, 0)
    middle = _sample_middle_transcript(parts, budget=middle_budget)
    marker = (
        f"\n\n[transcript truncated for brief — kept first {head_chars} chars, "
        f"{MIDDLE_SAMPLE_COUNT} representative middle sample(s), and last "
        f"{tail_chars} chars]\n\n"
    )
    ending_marker = "\n\n[ending excerpt]\n\n" if tail else ""
    if middle:
        middle = f"[representative middle excerpts]\n{middle}\n"
    return head + marker + middle + ending_marker + tail, True


def _sample_middle_transcript(parts: list[str], *, budget: int) -> str:
    if budget <= 0 or len(parts) < 3:
        return ""

    sample_indices: list[int] = []
    last_index = len(parts) - 1
    for offset in range(1, MIDDLE_SAMPLE_COUNT + 1):
        index = round(last_index * offset / (MIDDLE_SAMPLE_COUNT + 1))
        if index <= 0 or index >= last_index or index in sample_indices:
            continue
        sample_indices.append(index)
    if not sample_indices:
        return ""

    per_sample_budget = max(budget // len(sample_indices), 1)
    samples: list[str] = []
    for index in sample_indices:
        line = parts[index]
        if len(line) > per_sample_budget:
            line = line[: max(per_sample_budget - 1, 1)].rstrip() + "…"
        samples.append(line)
    combined = "\n".join(samples)
    return combined[:budget].rstrip()


def build_brief_request(
    *,
    job_id: str,
    video_path: Path,
    transcript_timeline: TranscriptTimeline,
    rubric_version: str = BRIEF_VERSION,
) -> VideoBriefRequest:
    excerpt, truncated = build_transcript_excerpt(transcript_timeline)
    duration = (
        transcript_timeline.segments[-1].end_seconds
        if transcript_timeline.segments
        else 0.0
    )
    speakers = _ordered_unique_speakers(transcript_timeline)
    return VideoBriefRequest(
        rubric_version=rubric_version,
        job_id=job_id,
        video_path=video_path,
        transcript_excerpt=excerpt,
        transcript_truncated=truncated,
        duration_seconds=duration,
        speakers=speakers,
        brief_prompt=build_brief_prompt(),
        response_schema=build_brief_response_schema(),
    )


def write_brief_request(
    workspace_dir: Path, request: VideoBriefRequest
) -> Path:
    path = brief_request_path(workspace_dir)
    write_json(path, request.model_dump(mode="json"))
    return path


def load_brief_request(workspace_dir: Path) -> VideoBriefRequest | None:
    path = brief_request_path(workspace_dir)
    if not path.exists():
        return None
    try:
        data = read_json(path)
    except (json.JSONDecodeError, OSError):
        return None
    try:
        return VideoBriefRequest.model_validate(data)
    except ValidationError:
        return None


def load_brief_response(workspace_dir: Path) -> VideoBriefResponse | None:
    path = brief_response_path(workspace_dir)
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BriefResponseError(
            f"Unable to read {BRIEF_RESPONSE_FILENAME}"
        ) from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BriefResponseError(
            f"{BRIEF_RESPONSE_FILENAME} is not valid JSON"
        ) from exc
    try:
        return VideoBriefResponse.model_validate(data)
    except ValidationError as exc:
        raise BriefResponseError(
            f"{BRIEF_RESPONSE_FILENAME} does not match brief contract"
        ) from exc


def persist_cached_brief(workspace_dir: Path, brief: VideoBrief) -> Path:
    """Save the last good brief so reruns don't require re-handoff.

    Mirrors scoring's per-clip cache: once the harness has authored a
    valid brief, cache it. On the next `advance` call, if no fresh
    brief-response.json exists, the cached one is reused as long as the
    rubric_version + job_id still match.
    """
    path = brief_cache_path(workspace_dir)
    write_json(path, brief.model_dump(mode="json"))
    return path


def load_cached_brief(workspace_dir: Path) -> VideoBrief | None:
    path = brief_cache_path(workspace_dir)
    if not path.exists():
        return None
    try:
        data = read_json(path)
    except (json.JSONDecodeError, OSError):
        return None
    try:
        return VideoBrief.model_validate(data)
    except ValidationError:
        return None


def resolve_brief(workspace_dir: Path) -> VideoBrief | None:
    """Return the brief for this workspace, or None if not yet authored.

    Resolution order:
    1. brief-response.json (validates rubric_version + job_id against
       brief-request.json) — the freshest source of truth.
    2. brief-cache.json (last good brief from a previous turn) — handles
       the common case where the harness already authored the brief and
       just hasn't re-emitted the response file on this turn.
    3. None — caller should emit the brief handoff and wait.
    """
    request = load_brief_request(workspace_dir)
    if request is None:
        return None

    try:
        response = load_brief_response(workspace_dir)
    except BriefResponseError:
        # Surface invalid responses; don't silently fall back to cache
        # on what may be a real authoring bug. The orchestrator/hermes
        # caller decides how to present this.
        raise

    if response is not None:
        if response.rubric_version != request.rubric_version:
            raise BriefResponseError(
                f"{BRIEF_RESPONSE_FILENAME} rubric_version "
                f"{response.rubric_version!r} != request "
                f"{request.rubric_version!r}"
            )
        if response.job_id != request.job_id:
            raise BriefResponseError(
                f"{BRIEF_RESPONSE_FILENAME} job_id "
                f"{response.job_id!r} != request {request.job_id!r}"
            )
        if response.brief.job_id != request.job_id:
            raise BriefResponseError(
                f"{BRIEF_RESPONSE_FILENAME} brief.job_id mismatch"
            )
        persist_cached_brief(workspace_dir, response.brief)
        return response.brief

    cached = load_cached_brief(workspace_dir)
    if cached is None:
        return None
    if cached.rubric_version != request.rubric_version:
        # Old cache from a prior rubric version — invalidate and force
        # the harness to re-author against the current contract.
        return None
    if cached.job_id != request.job_id:
        return None
    return cached


def _ordered_unique_speakers(timeline: TranscriptTimeline) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for segment in timeline.segments:
        speaker = segment.speaker
        if not speaker or speaker in seen:
            continue
        seen.add(speaker)
        ordered.append(speaker)
    return ordered
