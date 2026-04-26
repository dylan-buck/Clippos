from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

from pydantic import Field, Strict, StrictBool

from clippos.adapters import vision as vision_adapter
from clippos.adapters.storage import read_json, write_json
from clippos.models.media import ContractModel
from clippos.pipeline.fingerprint import compute_video_fingerprint

VISION_CACHE_FILENAME = "vision.json"

_LOGGER = logging.getLogger(__name__)


class FaceBox(ContractModel):
    center_x: Annotated[float, Strict(), Field(ge=0, le=1)]
    center_y: Annotated[float, Strict(), Field(ge=0, le=1)]
    width: Annotated[float, Strict(), Field(ge=0, le=1)]
    height: Annotated[float, Strict(), Field(ge=0, le=1)]
    confidence: Annotated[float, Strict(), Field(ge=0, le=1)]


class VisionFrame(ContractModel):
    timestamp_seconds: Annotated[float, Strict(), Field(ge=0)]
    motion_score: Annotated[float, Strict(), Field(ge=0, le=1)]
    shot_change: StrictBool
    primary_face: FaceBox | None = None


class VisionTimeline(ContractModel):
    frames: list[VisionFrame]


class VisionPayload(ContractModel):
    frames: list[VisionFrame]


def build_vision_timeline(payload: dict) -> VisionTimeline:
    validated_payload = VisionPayload.model_validate(payload)
    return VisionTimeline(frames=validated_payload.frames)


def run_vision(
    video_path: Path,
    workspace_dir: Path,
    *,
    config: vision_adapter.VisionConfig | None = None,
) -> dict:
    effective_config = config or vision_adapter.VisionConfig()
    cache_path = workspace_dir / VISION_CACHE_FILENAME
    source_fingerprint = compute_video_fingerprint(video_path)
    cached = _load_cached_vision(
        cache_path,
        model=vision_adapter.DEFAULT_MODEL,
        source_fingerprint=source_fingerprint,
    )
    if cached is not None:
        return cached

    result = vision_adapter.analyze(video_path, config=effective_config)
    payload = {"frames": result["frames"]}
    write_json(
        cache_path,
        {
            "metadata": {
                "model": result["model"],
                # Embed the fingerprint so a later run against a mutated
                # source file invalidates this cache instead of serving
                # stale face / motion frames.
                "source_fingerprint": source_fingerprint,
            },
            "payload": payload,
        },
    )
    return payload


def _load_cached_vision(
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
        # Pre-fingerprint cache. Treat as miss + warn rather than risk
        # serving stale frames against a mutated source file.
        _LOGGER.warning(
            "Vision cache at %s predates source-fingerprinting; ignoring "
            "and re-running vision analysis.",
            cache_path,
        )
        return None
    if cached_fingerprint != source_fingerprint:
        _LOGGER.warning(
            "Vision cache at %s was computed against a different source "
            "fingerprint (cached=%s, current=%s); re-running vision analysis.",
            cache_path,
            cached_fingerprint,
            source_fingerprint,
        )
        return None
    if "frames" not in payload:
        return None
    return payload
