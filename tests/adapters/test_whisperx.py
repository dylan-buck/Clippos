from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from clippos.adapters import whisperx as whisperx_adapter
from clippos.adapters.whisperx import (
    DIARIZER_OFF,
    DIARIZER_PYANNOTE,
    DIARIZER_SPEECHBRAIN,
    TranscriptionConfig,
    TranscriptionError,
    default_compute_type,
    detect_device,
    normalize_result,
    resolve_diarizer,
    resolve_hf_token,
    transcribe,
)
from clippos.pipeline.transcribe import build_transcript_timeline


@pytest.fixture(autouse=True)
def _clear_diarizer_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "CLIPPOS_DIARIZER",
    ):
        monkeypatch.delenv(name, raising=False)


def test_resolve_hf_token_prefers_primary_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "primary")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "secondary")

    assert resolve_hf_token() == "primary"


def test_resolve_hf_token_falls_back_to_alternate_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "fallback")

    assert resolve_hf_token() == "fallback"


def test_resolve_hf_token_raises_when_missing() -> None:
    with pytest.raises(TranscriptionError) as excinfo:
        resolve_hf_token()

    assert "Hugging Face token" in str(excinfo.value)
    assert "pyannote/speaker-diarization-3.1" in str(excinfo.value)


def test_detect_device_returns_cpu_without_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    assert detect_device() == "cpu"


def test_detect_device_returns_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert detect_device() == "cuda"


def test_detect_device_falls_back_to_cpu_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert detect_device() == "cpu"


def test_default_compute_type_uses_float16_on_cuda() -> None:
    assert default_compute_type("cuda") == "float16"


def test_default_compute_type_uses_int8_on_cpu() -> None:
    assert default_compute_type("cpu") == "int8"


def test_normalize_result_renames_word_keys_and_preserves_speakers() -> None:
    raw = {
        "segments": [
            {
                "start": 0.0,
                "end": 3.2,
                "text": " Look at this chart.",
                "speaker": "SPEAKER_00",
                "words": [
                    {
                        "word": "Look",
                        "start": 0.0,
                        "end": 0.28,
                        "score": 0.99,
                        "speaker": "SPEAKER_00",
                    },
                    {
                        "word": "at",
                        "start": 0.29,
                        "end": 0.42,
                        "score": 0.98,
                        "speaker": "SPEAKER_00",
                    },
                    {
                        "word": "this",
                        "start": 0.43,
                        "end": 0.74,
                        "score": 0.97,
                        "speaker": "SPEAKER_00",
                    },
                    {
                        "word": "chart.",
                        "start": 0.75,
                        "end": 1.4,
                        "score": 0.96,
                        "speaker": "SPEAKER_00",
                    },
                ],
            },
            {
                "start": 1.5,
                "end": 3.0,
                "text": " It really spikes.",
                "speaker": "SPEAKER_01",
                "words": [
                    {
                        "word": "It",
                        "start": 1.5,
                        "end": 1.66,
                        "score": 0.95,
                        "speaker": "SPEAKER_01",
                    },
                    {
                        "word": "really",
                        "start": 1.67,
                        "end": 2.04,
                        "score": 0.94,
                        "speaker": "SPEAKER_01",
                    },
                    {
                        "word": "spikes.",
                        "start": 2.05,
                        "end": 3.0,
                        "score": 0.93,
                        "speaker": "SPEAKER_01",
                    },
                ],
            },
        ]
    }

    payload = normalize_result(raw, model="large-v3", language="en")

    assert payload["model"] == "large-v3"
    assert payload["language"] == "en"
    assert [segment["speaker"] for segment in payload["segments"]] == [
        "SPEAKER_00",
        "SPEAKER_01",
    ]
    assert payload["segments"][0]["text"] == "Look at this chart."
    first_word = payload["segments"][0]["words"][0]
    assert first_word == {
        "text": "Look",
        "start_seconds": 0.0,
        "end_seconds": 0.28,
        "confidence": 0.99,
    }


def test_normalize_result_output_round_trips_into_transcript_timeline() -> None:
    raw = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.4,
                "text": "Look at this chart.",
                "speaker": "SPEAKER_00",
                "words": [
                    {
                        "word": "Look",
                        "start": 0.0,
                        "end": 0.28,
                        "score": 0.99,
                        "speaker": "SPEAKER_00",
                    },
                    {
                        "word": "chart.",
                        "start": 0.75,
                        "end": 1.4,
                        "score": 0.96,
                        "speaker": "SPEAKER_00",
                    },
                ],
            }
        ]
    }

    payload = normalize_result(raw, model="large-v3", language="en")
    timeline = build_transcript_timeline({"segments": payload["segments"]})

    assert timeline.segments[0].words[0].text == "Look"
    assert timeline.segments[0].speaker == "SPEAKER_00"


def test_normalize_result_clamps_confidence_out_of_range() -> None:
    raw = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "Hi",
                "speaker": "SPEAKER_00",
                "words": [
                    {
                        "word": "Hi",
                        "start": 0.0,
                        "end": 0.5,
                        "score": 1.5,
                        "speaker": "SPEAKER_00",
                    },
                    {
                        "word": "There",
                        "start": 0.5,
                        "end": 1.0,
                        "score": -0.2,
                        "speaker": "SPEAKER_00",
                    },
                ],
            }
        ]
    }

    payload = normalize_result(raw, model="large-v3", language="en")
    confidences = [word["confidence"] for word in payload["segments"][0]["words"]]

    assert confidences == [1.0, 0.0]


def test_normalize_result_replaces_nan_confidence_with_zero() -> None:
    raw = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "Hi",
                "speaker": "SPEAKER_00",
                "words": [
                    {
                        "word": "Hi",
                        "start": 0.0,
                        "end": 0.5,
                        "score": math.nan,
                        "speaker": "SPEAKER_00",
                    },
                ],
            }
        ]
    }

    payload = normalize_result(raw, model="large-v3", language="en")

    assert payload["segments"][0]["words"][0]["confidence"] == 0.0


def test_normalize_result_skips_words_missing_timestamps() -> None:
    raw = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Hello world",
                "speaker": "SPEAKER_00",
                "words": [
                    {
                        "word": "Hello",
                        "start": 0.0,
                        "end": 0.5,
                        "score": 0.9,
                        "speaker": "SPEAKER_00",
                    },
                    {"word": "world", "score": 0.8, "speaker": "SPEAKER_00"},
                ],
            }
        ]
    }

    payload = normalize_result(raw, model="large-v3", language="en")
    words = payload["segments"][0]["words"]

    assert len(words) == 1
    assert words[0]["text"] == "Hello"


def test_normalize_result_drops_segments_with_no_valid_words() -> None:
    raw = {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Empty",
                "speaker": "SPEAKER_00",
                "words": [{"word": "", "start": 0.0, "end": 0.1, "score": 0.9}],
            },
            {
                "start": 2.0,
                "end": 3.0,
                "text": "Valid",
                "speaker": "SPEAKER_01",
                "words": [
                    {
                        "word": "Valid",
                        "start": 2.0,
                        "end": 3.0,
                        "score": 0.9,
                        "speaker": "SPEAKER_01",
                    }
                ],
            },
        ]
    }

    payload = normalize_result(raw, model="large-v3", language="en")

    assert [segment["speaker"] for segment in payload["segments"]] == ["SPEAKER_01"]


def test_normalize_result_falls_back_to_word_speaker_when_segment_lacks_one() -> None:
    raw = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "Fallback",
                "words": [
                    {
                        "word": "Fallback",
                        "start": 0.0,
                        "end": 1.0,
                        "score": 0.9,
                        "speaker": "SPEAKER_02",
                    },
                ],
            }
        ]
    }

    payload = normalize_result(raw, model="large-v3", language="en")

    assert payload["segments"][0]["speaker"] == "SPEAKER_02"


def test_normalize_result_uses_unknown_when_no_speaker_anywhere() -> None:
    raw = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "No speaker",
                "words": [{"word": "No", "start": 0.0, "end": 0.5, "score": 0.9}],
            }
        ]
    }

    payload = normalize_result(raw, model="large-v3", language="en")

    assert payload["segments"][0]["speaker"] == "unknown"


def test_transcribe_composes_whisperx_and_pyannote_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.setenv("CLIPPOS_DIARIZER", DIARIZER_PYANNOTE)
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")

    calls: dict[str, object] = {}

    class FakeAsr:
        def transcribe(self, audio, batch_size):
            calls["asr.batch_size"] = batch_size
            calls["asr.audio"] = audio
            return {
                "language": "en",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 0.5,
                        "text": "Hi",
                        "words": [
                            {"word": "Hi", "start": 0.0, "end": 0.5, "score": 0.9}
                        ],
                    }
                ],
            }

    class FakeDiarizer:
        def __init__(self, use_auth_token, device):
            calls["diarizer.token"] = use_auth_token
            calls["diarizer.device"] = device

        def __call__(self, audio):
            calls["diarizer.audio"] = audio
            return "diarize-df"

    def fake_load_audio(path):
        calls["load_audio.path"] = path
        return "audio-array"

    def fake_load_model(model, device, compute_type):
        calls["asr.model"] = model
        calls["asr.device"] = device
        calls["asr.compute_type"] = compute_type
        return FakeAsr()

    def fake_load_align_model(language_code, device):
        calls["align.language"] = language_code
        calls["align.device"] = device
        return ("aligner", {"metadata": True})

    def fake_align(segments, aligner, metadata, audio, device, return_char_alignments):
        calls["align.segments"] = segments
        calls["align.return_char_alignments"] = return_char_alignments
        return {
            "segments": [
                {
                    "start": 0.0,
                    "end": 0.5,
                    "text": "Hi",
                    "words": [{"word": "Hi", "start": 0.0, "end": 0.5, "score": 0.9}],
                }
            ]
        }

    def fake_assign_word_speakers(diarize_df, aligned):
        calls["assign.diarize_df"] = diarize_df
        aligned["segments"][0]["speaker"] = "SPEAKER_00"
        aligned["segments"][0]["words"][0]["speaker"] = "SPEAKER_00"
        return aligned

    fake_whisperx = SimpleNamespace(
        load_audio=fake_load_audio,
        load_model=fake_load_model,
        load_align_model=fake_load_align_model,
        align=fake_align,
        DiarizationPipeline=FakeDiarizer,
        assign_word_speakers=fake_assign_word_speakers,
    )
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    result = transcribe(
        video,
        config=TranscriptionConfig(device="cpu", compute_type="int8", batch_size=4),
    )

    assert calls["load_audio.path"] == str(video)
    assert calls["asr.model"] == "large-v3"
    assert calls["asr.device"] == "cpu"
    assert calls["asr.compute_type"] == "int8"
    assert calls["asr.batch_size"] == 4
    assert calls["diarizer.token"] == "hf-token"
    assert calls["diarizer.device"] == "cpu"
    assert calls["diarizer.audio"] == "audio-array"
    assert result["model"] == "large-v3"
    assert result["language"] == "en"
    assert result["segments"][0]["speaker"] == "SPEAKER_00"
    assert result["segments"][0]["words"][0]["text"] == "Hi"


def test_transcribe_pyannote_path_refuses_without_hf_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The pyannote path still requires HF_TOKEN.

    The default path is now speechbrain (zero-config), so a missing token is
    only an error when the user explicitly opts into pyannote.
    """
    monkeypatch.setenv("CLIPPOS_DIARIZER", DIARIZER_PYANNOTE)
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")

    fake_whisperx = _build_fake_whisperx({}, raise_assign=False)
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    with pytest.raises(TranscriptionError) as excinfo:
        transcribe(video)
    # The error must mention HF_TOKEN AND point at the env var so users know
    # they can switch back to the open-source default.
    message = str(excinfo.value)
    assert "Hugging Face token" in message
    assert "CLIPPOS_DIARIZER" in message


def test_transcribe_default_path_uses_speechbrain_diarizer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No env vars set → speechbrain runs, pyannote is never touched."""
    pd = pytest.importorskip("pandas")
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")

    diarize_calls: list[object] = []

    def fake_diarize_audio(audio):
        diarize_calls.append(audio)
        return pd.DataFrame(
            [{"start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"}]
        )

    # Patch the attribute on the real (already-loaded) module so suite order
    # doesn't matter — `monkeypatch.setitem(sys.modules, ...)` only takes
    # effect when the module hasn't been imported yet, which other tests
    # in the suite may already have done.
    from clippos.adapters import speechbrain_diarize as _sd

    monkeypatch.setattr(_sd, "diarize_audio", fake_diarize_audio)

    pyannote_called: list[bool] = []

    class ExplodingDiarizer:
        def __init__(self, *args, **kwargs):
            pyannote_called.append(True)

        def __call__(self, audio):
            raise AssertionError("pyannote must not be called on the default path")

    fake_whisperx = _build_fake_whisperx({}, ExplodingDiarizer=ExplodingDiarizer)
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    result = transcribe(video, config=TranscriptionConfig(device="cpu", compute_type="int8"))

    assert diarize_calls == ["audio-array"]
    assert pyannote_called == []
    assert result["segments"][0]["speaker"] == "SPEAKER_00"


def test_transcribe_pyannote_path_falls_back_to_default_speaker_on_empty_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pyannote returns an empty DataFrame for fully-silent audio. Without
    a guard, segments end up speakerless. The fallback path must apply
    the same default speaker the speechbrain-empty path uses."""
    pd = pytest.importorskip("pandas")
    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.setenv("CLIPPOS_DIARIZER", DIARIZER_PYANNOTE)
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")

    class EmptyDiarizer:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, audio):
            return pd.DataFrame(columns=["start", "end", "speaker"])

    def assign_must_not_run(diarize_df, aligned):
        raise AssertionError(
            "assign_word_speakers must not run on the empty-result path"
        )

    fake_whisperx = SimpleNamespace(
        load_audio=lambda _path: "audio-array",
        load_model=lambda *_a, **_k: SimpleNamespace(
            transcribe=lambda audio, batch_size: {
                "language": "en",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 0.5,
                        "text": "Hi",
                        "words": [
                            {"word": "Hi", "start": 0.0, "end": 0.5, "score": 0.9}
                        ],
                    }
                ],
            }
        ),
        load_align_model=lambda **_k: ("aligner", {}),
        align=lambda *_a, **_k: {
            "segments": [
                {
                    "start": 0.0,
                    "end": 0.5,
                    "text": "Hi",
                    "words": [{"word": "Hi", "start": 0.0, "end": 0.5, "score": 0.9}],
                }
            ]
        },
        DiarizationPipeline=EmptyDiarizer,
        assign_word_speakers=assign_must_not_run,
    )
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    result = transcribe(
        video, config=TranscriptionConfig(device="cpu", compute_type="int8")
    )

    assert result["segments"][0]["speaker"] == whisperx_adapter.DEFAULT_FALLBACK_SPEAKER


def test_transcribe_off_path_stamps_fallback_speaker_without_diarizing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLIPPOS_DIARIZER=off → no diarizer touched, every segment gets the
    fallback speaker so downstream code can rely on the field."""
    monkeypatch.setenv("CLIPPOS_DIARIZER", DIARIZER_OFF)
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")

    def explode_speechbrain(audio):
        raise AssertionError("speechbrain must not run when diarizer is off")

    from clippos.adapters import speechbrain_diarize as _sd

    monkeypatch.setattr(_sd, "diarize_audio", explode_speechbrain)

    class ExplodingDiarizer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("pyannote must not run when diarizer is off")

    fake_whisperx = _build_fake_whisperx({}, ExplodingDiarizer=ExplodingDiarizer)
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)

    result = transcribe(video, config=TranscriptionConfig(device="cpu", compute_type="int8"))

    assert result["segments"][0]["speaker"] == whisperx_adapter.DEFAULT_FALLBACK_SPEAKER


def test_resolve_diarizer_defaults_to_speechbrain() -> None:
    assert resolve_diarizer() == DIARIZER_SPEECHBRAIN


def test_resolve_diarizer_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLIPPOS_DIARIZER", "pyannote")
    assert resolve_diarizer() == DIARIZER_PYANNOTE


def test_resolve_diarizer_explicit_arg_wins_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLIPPOS_DIARIZER", "pyannote")
    assert resolve_diarizer("off") == DIARIZER_OFF


def test_resolve_diarizer_rejects_unknown_value() -> None:
    with pytest.raises(TranscriptionError, match="Unsupported diarizer"):
        resolve_diarizer("magic")


def test_transcription_config_defaults_match_quality_first_choices() -> None:
    config = TranscriptionConfig()

    assert config.model == whisperx_adapter.DEFAULT_MODEL == "large-v3"
    assert config.device is None
    assert config.compute_type is None
    assert config.batch_size == 16
    assert config.diarizer is None  # falls through to env var / default


def _build_fake_whisperx(
    calls: dict[str, object],
    *,
    ExplodingDiarizer: type | None = None,
    raise_assign: bool = False,
) -> SimpleNamespace:
    """Compact helper that stitches a fake whisperx module for diarizer tests.

    The pyannote-path test still uses its own bespoke fakes to assert
    arguments; this helper covers the simpler default/off cases where we
    only care that ASR + alignment run and a diarizer either runs or doesn't.
    """

    class FakeAsr:
        def transcribe(self, audio, batch_size):
            return {
                "language": "en",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 0.5,
                        "text": "Hi",
                        "words": [
                            {"word": "Hi", "start": 0.0, "end": 0.5, "score": 0.9}
                        ],
                    }
                ],
            }

    def fake_load_audio(_path):
        return "audio-array"

    def fake_load_model(_model, device, compute_type):
        return FakeAsr()

    def fake_load_align_model(language_code, device):
        return ("aligner", {"metadata": True})

    def fake_align(segments, *_args, **_kwargs):
        return {
            "segments": [
                {
                    "start": 0.0,
                    "end": 0.5,
                    "text": "Hi",
                    "words": [{"word": "Hi", "start": 0.0, "end": 0.5, "score": 0.9}],
                }
            ]
        }

    def fake_assign_word_speakers(diarize_df, aligned):
        if raise_assign:
            raise AssertionError("assign_word_speakers must not run on this path")
        # Apply first row's speaker to the whole transcript so tests have a
        # deterministic value to assert on.
        if hasattr(diarize_df, "iloc") and len(diarize_df) > 0:
            speaker = diarize_df.iloc[0]["speaker"]
        else:
            speaker = whisperx_adapter.DEFAULT_FALLBACK_SPEAKER
        for segment in aligned["segments"]:
            segment["speaker"] = speaker
            for word in segment["words"]:
                word["speaker"] = speaker
        return aligned

    return SimpleNamespace(
        load_audio=fake_load_audio,
        load_model=fake_load_model,
        load_align_model=fake_load_align_model,
        align=fake_align,
        DiarizationPipeline=ExplodingDiarizer,
        assign_word_speakers=fake_assign_word_speakers,
    )
