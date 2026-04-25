from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "large-v3"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")

DIARIZER_SPEECHBRAIN = "speechbrain"
DIARIZER_PYANNOTE = "pyannote"
DIARIZER_OFF = "off"
DEFAULT_DIARIZER = DIARIZER_SPEECHBRAIN
VALID_DIARIZERS: tuple[str, ...] = (
    DIARIZER_SPEECHBRAIN,
    DIARIZER_PYANNOTE,
    DIARIZER_OFF,
)
_DIARIZER_ENV_VAR = "CLIPPOS_DIARIZER"
DEFAULT_FALLBACK_SPEAKER = "SPEAKER_00"


class TranscriptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class TranscriptionConfig:
    model: str = DEFAULT_MODEL
    device: str | None = None
    compute_type: str | None = None
    batch_size: int = 16
    diarizer: str | None = None  # falls back to env var or DEFAULT_DIARIZER


def resolve_diarizer(explicit: str | None = None) -> str:
    """Resolve which diarizer to use, defaulting to the open-source path.

    Priority: explicit arg → ``CLIPPOS_DIARIZER`` env var → default.
    """
    raw = (explicit or os.environ.get(_DIARIZER_ENV_VAR) or DEFAULT_DIARIZER).strip().lower()
    if raw not in VALID_DIARIZERS:
        raise TranscriptionError(
            f"Unsupported diarizer {raw!r}; expected one of: "
            f"{', '.join(VALID_DIARIZERS)}"
        )
    return raw


def resolve_hf_token() -> str:
    for var in _HF_TOKEN_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value
    raise TranscriptionError(
        "pyannote diarization requires a Hugging Face token. Set HF_TOKEN and "
        f"accept the {DIARIZATION_MODEL} license at "
        f"https://hf.co/{DIARIZATION_MODEL} before the first run, or unset "
        f"{_DIARIZER_ENV_VAR} to use the default open-source diarizer."
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
    diarizer_choice = resolve_diarizer(cfg.diarizer)

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

    merged = _apply_diarization(
        aligned=aligned,
        audio=audio,
        device=device,
        diarizer_choice=diarizer_choice,
        whisperx_module=whisperx,
    )
    return normalize_result(merged, model=cfg.model, language=language)


def _apply_diarization(
    *,
    aligned: dict,
    audio: Any,
    device: str,
    diarizer_choice: str,
    whisperx_module: Any,
) -> dict:
    """Attach per-segment speaker labels using the configured diarizer.

    - ``speechbrain`` (default): zero-config, uses silero-VAD + ECAPA-TDNN.
    - ``pyannote``: requires HF_TOKEN + license acceptance for the
      ``pyannote/speaker-diarization-3.1`` model.
    - ``off``: skip diarization; every segment gets a single fallback speaker
      so downstream code can rely on the field being non-null.
    """
    if diarizer_choice == DIARIZER_OFF:
        return _stamp_fallback_speaker(aligned, DEFAULT_FALLBACK_SPEAKER)

    if diarizer_choice == DIARIZER_PYANNOTE:
        hf_token = resolve_hf_token()
        diarizer = whisperx_module.DiarizationPipeline(
            use_auth_token=hf_token, device=device
        )
        diarize_segments = diarizer(audio)
        # Pyannote returns an empty DataFrame for fully-silent audio. Without
        # this guard, segments would end up with a missing `speaker` field
        # and downstream code (mining, normalization) would treat them as
        # unknown rather than a single fallback speaker.
        if _diarization_is_empty(diarize_segments):
            return _stamp_fallback_speaker(aligned, DEFAULT_FALLBACK_SPEAKER)
        return whisperx_module.assign_word_speakers(diarize_segments, aligned)

    # speechbrain (default, zero-config).
    from clippos.adapters import speechbrain_diarize

    diarize_df = speechbrain_diarize.diarize_audio(audio)
    if _diarization_is_empty(diarize_df):
        return _stamp_fallback_speaker(aligned, DEFAULT_FALLBACK_SPEAKER)
    return whisperx_module.assign_word_speakers(diarize_df, aligned)


def _diarization_is_empty(diarize_result: Any) -> bool:
    """True when the diarizer returned no usable speaker segments.

    Both pyannote and the open-source diarizer return a pandas DataFrame
    in the happy path; this helper tolerates ``None``, empty DataFrames,
    and empty-list-likes so future diarizer plugins don't have to also
    return a DataFrame.
    """
    if diarize_result is None:
        return True
    try:
        return len(diarize_result) == 0
    except TypeError:
        return False


def _stamp_fallback_speaker(aligned: dict, speaker: str) -> dict:
    """Mark every segment + word with ``speaker`` so downstream code is unconditional."""
    for segment in aligned.get("segments", []) or []:
        if not isinstance(segment, dict):
            continue
        segment.setdefault("speaker", speaker)
        for word in segment.get("words", []) or []:
            if isinstance(word, dict):
                word.setdefault("speaker", speaker)
    return aligned


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
