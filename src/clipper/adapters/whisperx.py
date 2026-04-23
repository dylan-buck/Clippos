from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "large-v3"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")


class TranscriptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class TranscriptionConfig:
    model: str = DEFAULT_MODEL
    device: str | None = None
    compute_type: str | None = None
    batch_size: int = 16


def resolve_hf_token() -> str:
    for var in _HF_TOKEN_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value
    raise TranscriptionError(
        "Diarization requires a Hugging Face token. Set HF_TOKEN and accept the "
        f"{DIARIZATION_MODEL} license at https://hf.co/{DIARIZATION_MODEL} "
        "before the first run."
    )


def detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def default_compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


def transcribe(video_path: Path, *, config: TranscriptionConfig | None = None) -> dict:
    cfg = config or TranscriptionConfig()
    device = cfg.device or detect_device()
    compute_type = cfg.compute_type or default_compute_type(device)
    hf_token = resolve_hf_token()

    import whisperx

    audio = whisperx.load_audio(str(video_path))

    asr = whisperx.load_model(cfg.model, device=device, compute_type=compute_type)
    result = asr.transcribe(audio, batch_size=cfg.batch_size)
    language = result["language"]

    aligner, metadata = whisperx.load_align_model(language_code=language, device=device)
    aligned = whisperx.align(
        result["segments"],
        aligner,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    diarizer = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarizer(audio)
    merged = whisperx.assign_word_speakers(diarize_segments, aligned)

    return normalize_result(merged, model=cfg.model, language=language)


def normalize_result(raw: dict, *, model: str, language: str) -> dict:
    segments: list[dict[str, Any]] = []
    for raw_segment in raw.get("segments", []):
        words = _normalize_words(raw_segment.get("words", []))
        if not words:
            continue
        speaker = raw_segment.get("speaker") or words[0].get("_speaker") or "unknown"
        segment_start = _coerce_seconds(
            raw_segment.get("start"), fallback=words[0]["start_seconds"]
        )
        segment_end = _coerce_seconds(
            raw_segment.get("end"), fallback=words[-1]["end_seconds"]
        )
        if segment_end < segment_start:
            segment_end = segment_start
        text = str(raw_segment.get("text") or "").strip()
        segments.append(
            {
                "speaker": speaker,
                "start_seconds": segment_start,
                "end_seconds": segment_end,
                "text": text,
                "words": [_strip_internal_fields(word) for word in words],
            }
        )
    return {
        "model": model,
        "language": language,
        "segments": segments,
    }


def _normalize_words(raw_words: list[dict]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw_word in raw_words:
        text = raw_word.get("word") or raw_word.get("text")
        if text is None:
            continue
        text = str(text).strip()
        if not text:
            continue
        start = _coerce_seconds(raw_word.get("start"))
        end = _coerce_seconds(raw_word.get("end"))
        if start is None or end is None:
            continue
        if end < start:
            end = start
        normalized.append(
            {
                "text": text,
                "start_seconds": start,
                "end_seconds": end,
                "confidence": _coerce_confidence(raw_word.get("score")),
                "_speaker": raw_word.get("speaker"),
            }
        )
    return normalized


def _strip_internal_fields(word: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in word.items() if not key.startswith("_")}


def _coerce_seconds(value: Any, *, fallback: float | None = None) -> float | None:
    if value is None:
        return fallback
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(coerced) or math.isinf(coerced):
        return fallback
    return max(coerced, 0.0)


def _coerce_confidence(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(coerced) or math.isinf(coerced):
        return 0.0
    return min(max(coerced, 0.0), 1.0)
