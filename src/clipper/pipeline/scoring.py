from __future__ import annotations

import json
from hashlib import sha1
from pathlib import Path

from pydantic import ValidationError

from clipper.adapters.rubric import (
    RUBRIC_VERSION,
    build_response_schema,
    build_rubric_prompt,
)
from clipper.adapters.storage import read_json, write_json
from clipper.models.candidate import CandidateClip
from clipper.models.scoring import (
    ClipBrief,
    ClipScore,
    MiningSignals,
    ScoringRequest,
    ScoringResponse,
    VideoBrief,
)
from clipper.pipeline.candidates import ScoredWindow
from clipper.pipeline.transcribe import TranscriptSegment

SCORING_REQUEST_FILENAME = "scoring-request.json"
SCORING_RESPONSE_FILENAME = "scoring-response.json"
SCORING_CACHE_DIRNAME = "scoring-cache"


class ScoringResponseError(RuntimeError):
    pass


def compute_clip_hash(
    *,
    transcript: str,
    start_seconds: float,
    end_seconds: float,
    rubric_version: str,
) -> str:
    key = f"{rubric_version}|{start_seconds:.3f}|{end_seconds:.3f}|{transcript.strip()}"
    return sha1(key.encode("utf-8")).hexdigest()[:16]


def build_clip_brief(
    *,
    candidate: CandidateClip,
    scored: ScoredWindow,
    rubric_version: str = RUBRIC_VERSION,
) -> ClipBrief:
    transcript = " ".join(segment.text for segment in scored.segments).strip()
    speakers = _ordered_unique_speakers(scored.segments)
    duration = candidate.end_seconds - candidate.start_seconds
    clip_hash = compute_clip_hash(
        transcript=transcript,
        start_seconds=candidate.start_seconds,
        end_seconds=candidate.end_seconds,
        rubric_version=rubric_version,
    )
    signals = scored.signals
    mining_signals = MiningSignals(
        hook=signals.hook,
        keyword=signals.keyword,
        numeric=signals.numeric,
        interjection=signals.interjection,
        payoff=signals.payoff,
        question_to_answer=signals.question_to_answer,
        motion=signals.motion,
        shot_change=signals.shot_change,
        face_presence=signals.face_presence,
        speaker_interaction=signals.speaker_interaction,
        delivery_variance=signals.delivery_variance,
        buried_lead=signals.buried_lead,
        dangling_question=signals.dangling_question,
        rambling_middle=signals.rambling_middle,
        reasons=list(candidate.reasons),
        spike_categories=list(candidate.spike_categories),
    )
    return ClipBrief(
        clip_id=candidate.clip_id,
        clip_hash=clip_hash,
        start_seconds=candidate.start_seconds,
        end_seconds=candidate.end_seconds,
        duration_seconds=duration,
        transcript=transcript,
        speakers=speakers,
        mining_score=candidate.score,
        mining_signals=mining_signals,
    )


def build_scoring_request(
    *,
    job_id: str,
    video_path: Path,
    briefs: list[ClipBrief],
    rubric_version: str = RUBRIC_VERSION,
    video_brief: VideoBrief | None = None,
) -> ScoringRequest:
    return ScoringRequest(
        rubric_version=rubric_version,
        job_id=job_id,
        video_path=video_path,
        rubric_prompt=build_rubric_prompt(),
        response_schema=build_response_schema(),
        clips=briefs,
        video_brief=video_brief,
    )


def scoring_request_path(workspace_dir: Path) -> Path:
    return workspace_dir / SCORING_REQUEST_FILENAME


def scoring_response_path(workspace_dir: Path) -> Path:
    return workspace_dir / SCORING_RESPONSE_FILENAME


def scoring_cache_dir(workspace_dir: Path) -> Path:
    return workspace_dir / SCORING_CACHE_DIRNAME


def write_scoring_request(workspace_dir: Path, request: ScoringRequest) -> Path:
    path = scoring_request_path(workspace_dir)
    write_json(path, request.model_dump(mode="json"))
    return path


def load_scoring_request(workspace_dir: Path) -> ScoringRequest | None:
    path = scoring_request_path(workspace_dir)
    if not path.exists():
        return None
    try:
        data = read_json(path)
    except (json.JSONDecodeError, OSError):
        return None
    try:
        return ScoringRequest.model_validate(data)
    except ValidationError:
        return None


def load_scoring_response(workspace_dir: Path) -> ScoringResponse | None:
    path = scoring_response_path(workspace_dir)
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScoringResponseError(
            f"Unable to read {SCORING_RESPONSE_FILENAME}"
        ) from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ScoringResponseError(
            f"{SCORING_RESPONSE_FILENAME} is not valid JSON"
        ) from exc
    try:
        return ScoringResponse.model_validate(data)
    except ValidationError as exc:
        raise ScoringResponseError(
            f"{SCORING_RESPONSE_FILENAME} does not match scoring contract"
        ) from exc


def load_cached_score(workspace_dir: Path, clip_hash: str) -> ClipScore | None:
    path = scoring_cache_dir(workspace_dir) / f"{clip_hash}.json"
    if not path.exists():
        return None
    try:
        data = read_json(path)
    except (json.JSONDecodeError, OSError):
        return None
    try:
        return ClipScore.model_validate(data)
    except ValidationError:
        return None


def persist_cached_score(workspace_dir: Path, score: ClipScore) -> Path:
    path = scoring_cache_dir(workspace_dir) / f"{score.clip_hash}.json"
    write_json(path, score.model_dump(mode="json"))
    return path


def resolve_scores(workspace_dir: Path) -> list[ClipScore] | None:
    request = load_scoring_request(workspace_dir)
    if request is None:
        return None

    response = load_scoring_response(workspace_dir)
    if response is not None and response.rubric_version != request.rubric_version:
        raise ScoringResponseError(
            "scoring-response.json rubric_version "
            f"{response.rubric_version!r} != request {request.rubric_version!r}"
        )
    if response is not None and response.job_id != request.job_id:
        raise ScoringResponseError(
            f"scoring-response.json job_id {response.job_id!r} != request "
            f"{request.job_id!r}"
        )

    response_by_hash: dict[str, ClipScore] = {}
    if response is not None:
        for clip_score in response.scores:
            response_by_hash[clip_score.clip_hash] = clip_score

    resolved: list[ClipScore] = []
    for brief in request.clips:
        if brief.clip_hash in response_by_hash:
            score = response_by_hash[brief.clip_hash]
            if score.clip_id != brief.clip_id:
                raise ScoringResponseError(
                    f"scoring-response.json clip_hash {brief.clip_hash!r} "
                    f"returned clip_id {score.clip_id!r}, expected {brief.clip_id!r}"
                )
            resolved.append(score)
            persist_cached_score(workspace_dir, score)
            continue
        cached = load_cached_score(workspace_dir, brief.clip_hash)
        if cached is not None:
            resolved.append(cached.model_copy(update={"clip_id": brief.clip_id}))
            continue
        return None

    return resolved


def scores_to_model_payload(scores: list[ClipScore]) -> list[dict]:
    return [
        {
            "clip_id": score.clip_id,
            "title": score.title,
            "hook": score.hook,
            "reasons": list(score.reasons),
            "final_score": score.final_score,
        }
        for score in scores
    ]


def _ordered_unique_speakers(segments: tuple[TranscriptSegment, ...]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for segment in segments:
        if segment.speaker in seen:
            continue
        seen.add(segment.speaker)
        ordered.append(segment.speaker)
    return ordered
