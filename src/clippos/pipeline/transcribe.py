from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator

from clippos.adapters import whisperx as whisperx_adapter
from clippos.adapters.storage import read_json, write_json
from clippos.models.media import ContractModel
from clippos.pipeline.fingerprint import compute_video_fingerprint

TRANSCRIPT_CACHE_FILENAME = "transcript.json"

_LOGGER = logging.getLogger(__name__)


class TranscriptWord(ContractModel):
    text: str
    start_seconds: Annotated[float, Field(ge=0)]
    end_seconds: Annotated[float, Field(ge=0)]
    confidence: Annotated[float, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def validate_time_bounds(self) -> "TranscriptWord":
        if self.end_seconds < self.start_seconds:
            raise ValueError(
                "end_seconds must be greater than or equal to start_seconds"
            )
        return self


class TranscriptSegment(ContractModel):
    speaker: str
    start_seconds: Annotated[float, Field(ge=0)]
    end_seconds: Annotated[float, Field(ge=0)]
    text: str
    words: list[TranscriptWord]

    @model_validator(mode="after")
    def validate_time_bounds(self) -> "TranscriptSegment":
        if self.end_seconds < self.start_seconds:
            raise ValueError(
                "end_seconds must be greater than or equal to start_seconds"
            )
        return self


class TranscriptTimeline(ContractModel):
    segments: list[TranscriptSegment]


class TranscriptPayload(ContractModel):
    segments: list[TranscriptSegment]


def build_transcript_timeline(payload: dict) -> TranscriptTimeline:
    validated_payload = TranscriptPayload.model_validate(payload)
    return TranscriptTimeline(segments=validated_payload.segments)


def run_transcription(
    video_path: Path,
    workspace_dir: Path,
    *,
    config: whisperx_adapter.TranscriptionConfig | None = None,
) -> dict:
    effective_config = config or whisperx_adapter.TranscriptionConfig()
    cache_path = workspace_dir / TRANSCRIPT_CACHE_FILENAME
    source_fingerprint = compute_video_fingerprint(video_path)
    cached = _load_cached_transcript(
        cache_path,
        model=effective_config.model,
        source_fingerprint=source_fingerprint,
    )
    if cached is not None:
        return cached

    result = whisperx_adapter.transcribe(video_path, config=effective_config)
    payload = {"segments": result["segments"]}
    write_json(
        cache_path,
        {
            "metadata": {
                "model": result["model"],
                "language": result["language"],
                # Embed the fingerprint so a later run with a mutated
                # source file (size or mtime delta) can detect the
                # mismatch and force a re-mine instead of silently
                # serving a stale transcript.
                "source_fingerprint": source_fingerprint,
            },
            "payload": payload,
        },
    )
    return payload


def _load_cached_transcript(
    cache_path: Path, *, model: str, source_fingerprint: str
) -> dict | None:
    if not cache_path.exists():
        return None
    try:
        data = read_json(cache_path)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    metadata = data.get("metadata")
    payload = data.get("payload")
    if not isinstance(metadata, dict) or not isinstance(payload, dict):
        return None
    if metadata.get("model") != model:
        return None
    cached_fingerprint = metadata.get("source_fingerprint")
    if cached_fingerprint is None:
        # Pre-fingerprint cache (predates the workspace-fingerprint
        # change). Treat as a miss + warn so we re-mine against the
        # current content rather than risk serving stale segments.
        _LOGGER.warning(
            "Transcript cache at %s predates source-fingerprinting; "
            "ignoring and re-running transcription.",
            cache_path,
        )
        return None
    if cached_fingerprint != source_fingerprint:
        _LOGGER.warning(
            "Transcript cache at %s was computed against a different source "
            "fingerprint (cached=%s, current=%s); re-running transcription.",
            cache_path,
            cached_fingerprint,
            source_fingerprint,
        )
        return None
    if "segments" not in payload:
        return None
    return payload
