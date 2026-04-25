import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from clippos.adapters.whisperx import TranscriptionConfig
from clippos.pipeline.transcribe import (
    TRANSCRIPT_CACHE_FILENAME,
    build_transcript_timeline,
    run_transcription,
)


def test_build_transcript_timeline_returns_word_ranges(
    sample_transcript_payload: dict,
) -> None:
    timeline = build_transcript_timeline(sample_transcript_payload)

    assert timeline.segments[0].speaker == "speaker_1"
    assert timeline.segments[0].start_seconds == 0.0
    assert timeline.segments[0].words[0].text == "Look"
    assert timeline.segments[0].words[0].start_seconds == 0.0
    assert timeline.segments[0].words[0].end_seconds == 0.28


def test_build_transcript_timeline_keeps_speaker_turns(
    sample_transcript_payload: dict,
) -> None:
    timeline = build_transcript_timeline(sample_transcript_payload)

    assert [segment.speaker for segment in timeline.segments] == [
        "speaker_1",
        "speaker_2",
    ]


def test_build_transcript_timeline_allows_zero_length_word_and_segment() -> None:
    timeline = build_transcript_timeline(
        {
            "segments": [
                {
                    "speaker": "speaker_1",
                    "start_seconds": 0.0,
                    "end_seconds": 0.0,
                    "text": "Look",
                    "words": [
                        {
                            "text": "Look",
                            "start_seconds": 0.0,
                            "end_seconds": 0.0,
                            "confidence": 0.99,
                        }
                    ],
                }
            ]
        }
    )

    assert timeline.segments[0].end_seconds == 0.0
    assert timeline.segments[0].words[0].end_seconds == 0.0


def test_build_transcript_timeline_rejects_reversed_word_time_bounds(
    sample_transcript_payload: dict,
) -> None:
    sample_transcript_payload["segments"][0]["words"][0]["start_seconds"] = 0.5
    sample_transcript_payload["segments"][0]["words"][0]["end_seconds"] = 0.2

    with pytest.raises(ValidationError):
        build_transcript_timeline(sample_transcript_payload)


def test_build_transcript_timeline_rejects_negative_segment_times(
    sample_transcript_payload: dict,
) -> None:
    sample_transcript_payload["segments"][1]["end_seconds"] = -0.1

    with pytest.raises(ValidationError):
        build_transcript_timeline(sample_transcript_payload)


def test_build_transcript_timeline_rejects_reversed_segment_time_bounds(
    sample_transcript_payload: dict,
) -> None:
    sample_transcript_payload["segments"][1]["start_seconds"] = 3.1
    sample_transcript_payload["segments"][1]["end_seconds"] = 3.0

    with pytest.raises(ValidationError):
        build_transcript_timeline(sample_transcript_payload)


def test_build_transcript_timeline_rejects_malformed_payload_shape() -> None:
    with pytest.raises(ValidationError):
        build_transcript_timeline({"segments": [{"speaker": "speaker_1"}]})


def _fake_adapter_result(
    sample_transcript_payload: dict, *, model: str = "large-v3"
) -> dict:
    return {
        "model": model,
        "language": "en",
        "segments": sample_transcript_payload["segments"],
    }


def test_run_transcription_writes_cache_and_returns_payload(
    tmp_path: Path,
    sample_transcript_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls: list[Path] = []

    def fake_transcribe(path: Path, *, config: TranscriptionConfig) -> dict:
        calls.append(path)
        return _fake_adapter_result(sample_transcript_payload, model=config.model)

    monkeypatch.setattr(
        "clippos.pipeline.transcribe.whisperx_adapter.transcribe", fake_transcribe
    )

    payload = run_transcription(video, workspace)

    assert calls == [video]
    assert payload == {"segments": sample_transcript_payload["segments"]}
    cache_path = workspace / TRANSCRIPT_CACHE_FILENAME
    assert cache_path.exists()
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cached["metadata"] == {"model": "large-v3", "language": "en"}
    assert cached["payload"] == payload


def test_run_transcription_round_trips_into_timeline(
    tmp_path: Path,
    sample_transcript_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setattr(
        "clippos.pipeline.transcribe.whisperx_adapter.transcribe",
        lambda path, *, config: _fake_adapter_result(sample_transcript_payload),
    )

    payload = run_transcription(video, workspace)
    timeline = build_transcript_timeline(payload)

    assert [segment.speaker for segment in timeline.segments] == [
        "speaker_1",
        "speaker_2",
    ]


def test_run_transcription_reuses_cache_when_model_matches(
    tmp_path: Path,
    sample_transcript_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def explode(path: Path, *, config: TranscriptionConfig) -> dict:
        raise AssertionError("adapter should not be called on cache hit")

    cache_path = workspace / TRANSCRIPT_CACHE_FILENAME
    cache_path.write_text(
        json.dumps(
            {
                "metadata": {"model": "large-v3", "language": "en"},
                "payload": {"segments": sample_transcript_payload["segments"]},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "clippos.pipeline.transcribe.whisperx_adapter.transcribe", explode
    )

    payload = run_transcription(video, workspace)

    assert payload == {"segments": sample_transcript_payload["segments"]}


def test_run_transcription_reruns_when_cached_model_mismatches(
    tmp_path: Path,
    sample_transcript_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache_path = workspace / TRANSCRIPT_CACHE_FILENAME
    cache_path.write_text(
        json.dumps(
            {
                "metadata": {"model": "small", "language": "en"},
                "payload": {"segments": sample_transcript_payload["segments"]},
            }
        ),
        encoding="utf-8",
    )
    calls: list[Path] = []

    def fake_transcribe(path: Path, *, config: TranscriptionConfig) -> dict:
        calls.append(path)
        return _fake_adapter_result(sample_transcript_payload, model=config.model)

    monkeypatch.setattr(
        "clippos.pipeline.transcribe.whisperx_adapter.transcribe", fake_transcribe
    )

    payload = run_transcription(video, workspace)

    assert calls == [video]
    assert payload == {"segments": sample_transcript_payload["segments"]}
    refreshed = json.loads(cache_path.read_text(encoding="utf-8"))
    assert refreshed["metadata"]["model"] == "large-v3"


def test_run_transcription_reruns_when_cache_is_corrupt(
    tmp_path: Path,
    sample_transcript_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache_path = workspace / TRANSCRIPT_CACHE_FILENAME
    cache_path.write_text("not json", encoding="utf-8")
    calls: list[Path] = []

    def fake_transcribe(path: Path, *, config: TranscriptionConfig) -> dict:
        calls.append(path)
        return _fake_adapter_result(sample_transcript_payload, model=config.model)

    monkeypatch.setattr(
        "clippos.pipeline.transcribe.whisperx_adapter.transcribe", fake_transcribe
    )

    payload = run_transcription(video, workspace)

    assert calls == [video]
    assert payload == {"segments": sample_transcript_payload["segments"]}
    refreshed = json.loads(cache_path.read_text(encoding="utf-8"))
    assert refreshed["metadata"]["model"] == "large-v3"


def test_run_transcription_reruns_when_cache_shape_is_wrong(
    tmp_path: Path,
    sample_transcript_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache_path = workspace / TRANSCRIPT_CACHE_FILENAME
    cache_path.write_text(
        json.dumps({"segments": sample_transcript_payload["segments"]}),
        encoding="utf-8",
    )

    def fake_transcribe(path: Path, *, config: TranscriptionConfig) -> dict:
        return _fake_adapter_result(sample_transcript_payload, model=config.model)

    monkeypatch.setattr(
        "clippos.pipeline.transcribe.whisperx_adapter.transcribe", fake_transcribe
    )

    payload = run_transcription(video, workspace)

    assert payload == {"segments": sample_transcript_payload["segments"]}
    refreshed = json.loads(cache_path.read_text(encoding="utf-8"))
    assert "metadata" in refreshed and "payload" in refreshed
