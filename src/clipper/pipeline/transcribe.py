from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator

from clipper.adapters import whisperx as whisperx_adapter
from clipper.adapters.storage import read_json, write_json
from clipper.models.media import ContractModel

TRANSCRIPT_CACHE_FILENAME = "transcript.json"


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
    cached = _load_cached_transcript(cache_path, model=effective_config.model)
    if cached is not None:
        return cached

    result = whisperx_adapter.transcribe(video_path, config=effective_config)
    payload = {"segments": result["segments"]}
    write_json(
        cache_path,
        {
            "metadata": {"model": result["model"], "language": result["language"]},
            "payload": payload,
        },
    )
    return payload


def _load_cached_transcript(cache_path: Path, *, model: str) -> dict | None:
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
    if "segments" not in payload:
        return None
    return payload
