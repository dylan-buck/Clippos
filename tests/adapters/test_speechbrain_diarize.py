"""Tests for the open-source speaker diarization adapter.

The adapter has three responsibilities — VAD, embedding, clustering — plus
DataFrame emission. Clustering is pure-numpy so we test it directly with
synthetic embeddings. End-to-end behavior is exercised by stubbing the lazy
imports for ``silero_vad`` and ``speechbrain.inference.speaker``; otherwise
these tests would require ~80 MB of weights and an internet round-trip.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

# Diarization clusters embeddings + emits a pandas DataFrame; skip the whole
# module when those deps aren't installed (typical of the dev-only extra).
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from clipper.adapters import speechbrain_diarize as sd  # noqa: E402

# The end-to-end stubbed tests still need a real torch.from_numpy + no_grad
# context manager + Tensor methods, which is too much surface to stub.
# Clustering and DataFrame-shape tests run without torch.
_TORCH_AVAILABLE = True
try:  # noqa: SIM105
    import torch  # type: ignore  # noqa: F401
except ImportError:
    _TORCH_AVAILABLE = False
needs_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE,
    reason="torch is required for VAD/embedding integration tests; install engine extras",
)


# ---------- clustering primitives ----------


def test_cluster_embeddings_returns_empty_for_empty_matrix() -> None:
    empty = np.zeros((0, 192), dtype="float32")
    assert sd.cluster_embeddings(empty) == []


def test_cluster_embeddings_handles_single_segment() -> None:
    one = np.array([[1.0, 0.0, 0.0]], dtype="float32")
    assert sd.cluster_embeddings(one) == [0]


def test_cluster_embeddings_two_similar_segments_merge_to_single_speaker() -> None:
    embeddings = np.array(
        [
            [1.0, 0.05, 0.0],
            [1.0, 0.0, 0.05],
        ],
        dtype="float32",
    )
    labels = sd.cluster_embeddings(
        embeddings, two_segment_cosine_threshold=0.65
    )
    assert labels == [0, 0]


def test_cluster_embeddings_two_dissimilar_segments_split_to_two_speakers() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype="float32",
    )
    labels = sd.cluster_embeddings(
        embeddings, two_segment_cosine_threshold=0.65
    )
    assert labels == [0, 1]


def test_cluster_embeddings_recovers_two_distinct_speaker_clusters() -> None:
    rng = np.random.default_rng(42)
    speaker_a = rng.normal(loc=[3.0, 0.0, 0.0], scale=0.05, size=(8, 3))
    speaker_b = rng.normal(loc=[0.0, 3.0, 0.0], scale=0.05, size=(8, 3))
    embeddings = np.concatenate([speaker_a, speaker_b], axis=0).astype("float32")

    labels = sd.cluster_embeddings(embeddings, max_speakers=4)

    # Exactly two unique labels recovered, and the within-block labels are
    # consistent (renumbering by first appearance means block A is all 0,
    # block B is all 1).
    assert sorted(set(labels)) == [0, 1]
    assert all(label == labels[0] for label in labels[:8])
    assert all(label == labels[8] for label in labels[8:])
    assert labels[0] != labels[8]


def test_cluster_embeddings_renumbers_by_first_appearance() -> None:
    # Construct a case where spectral clustering may name clusters in an
    # arbitrary order. Renumbering should always start at 0 for the first
    # segment, regardless of the underlying cluster ids.
    rng = np.random.default_rng(7)
    cluster_a = rng.normal(loc=[5.0, 0.0], scale=0.02, size=(5, 2))
    cluster_b = rng.normal(loc=[0.0, 5.0], scale=0.02, size=(5, 2))
    embeddings = np.concatenate([cluster_a, cluster_b], axis=0).astype("float32")

    labels = sd.cluster_embeddings(embeddings, max_speakers=4)
    assert labels[0] == 0


# ---------- DataFrame schema ----------


def test_build_diarization_dataframe_emits_whisperx_compatible_columns() -> None:
    speech_timestamps = [
        {"start": 0.0, "end": 1.5},
        {"start": 1.7, "end": 3.2},
    ]
    df = sd._build_diarization_dataframe(speech_timestamps, [0, 1])

    assert list(df.columns) == ["start", "end", "speaker"]
    assert df.iloc[0]["speaker"] == "SPEAKER_00"
    assert df.iloc[1]["speaker"] == "SPEAKER_01"
    assert df.iloc[0]["start"] == pytest.approx(0.0)
    assert df.iloc[1]["end"] == pytest.approx(3.2)


def test_empty_diarization_dataframe_has_correct_schema() -> None:
    empty = sd._empty_diarization_dataframe(pd)
    assert list(empty.columns) == ["start", "end", "speaker"]
    assert len(empty) == 0


# ---------- end-to-end with stubbed imports ----------


@pytest.fixture
def stub_speechbrain_stack(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the lazy `silero_vad` and `speechbrain.inference.speaker`
    imports with deterministic in-memory fakes.

    Returns a handle the test can use to inspect what got called.
    """
    state: dict[str, Any] = {
        "vad_calls": [],
        "embed_calls": [],
        "vad_segments": [
            {"start": 0.0, "end": 1.5},
            {"start": 1.7, "end": 3.2},
            {"start": 3.4, "end": 5.0},
        ],
        # Two distinct speakers across three segments: A, B, A.
        "embeddings": np.array(
            [
                [3.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype="float32",
        ),
    }

    def fake_load_silero_vad() -> object:
        return object()

    def fake_get_speech_timestamps(
        waveform: Any, model: Any, **kwargs: Any
    ) -> list[dict[str, float]]:
        state["vad_calls"].append({"shape": tuple(waveform.shape), **kwargs})
        return state["vad_segments"]

    silero_module = types.ModuleType("silero_vad")
    silero_module.load_silero_vad = fake_load_silero_vad  # type: ignore[attr-defined]
    silero_module.get_speech_timestamps = fake_get_speech_timestamps  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "silero_vad", silero_module)

    class FakeEmbeddingTensor:
        def __init__(self, vector: np.ndarray) -> None:
            self._vector = vector

        def detach(self) -> "FakeEmbeddingTensor":
            return self

        def squeeze(self) -> "FakeEmbeddingTensor":
            return self

        def cpu(self) -> "FakeEmbeddingTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self._vector

    class FakeClassifier:
        def __init__(self) -> None:
            self.calls = 0

        def encode_batch(self, signal: Any) -> FakeEmbeddingTensor:
            vector = state["embeddings"][self.calls]
            self.calls += 1
            state["embed_calls"].append(tuple(signal.shape))
            return FakeEmbeddingTensor(vector)

    class FakeEncoderClassifier:
        @staticmethod
        def from_hparams(source: str, savedir: str | None = None) -> FakeClassifier:
            state["classifier_source"] = source
            state["classifier_savedir"] = savedir
            return FakeClassifier()

    speaker_module = types.ModuleType("speechbrain.inference.speaker")
    speaker_module.EncoderClassifier = FakeEncoderClassifier  # type: ignore[attr-defined]

    inference_module = types.ModuleType("speechbrain.inference")
    inference_module.speaker = speaker_module  # type: ignore[attr-defined]

    speechbrain_module = types.ModuleType("speechbrain")
    speechbrain_module.inference = inference_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "speechbrain", speechbrain_module)
    monkeypatch.setitem(sys.modules, "speechbrain.inference", inference_module)
    monkeypatch.setitem(sys.modules, "speechbrain.inference.speaker", speaker_module)

    return state


@needs_torch
def test_diarize_audio_end_to_end_with_stubbed_stack(
    stub_speechbrain_stack: dict[str, Any],
) -> None:
    sample_rate = sd.DEFAULT_SAMPLE_RATE
    audio = np.zeros(sample_rate * 6, dtype="float32")  # 6 seconds of silence
    # Insert non-silence so chunks aren't trimmed by the min-duration guard.
    audio[: sample_rate * 5] = 0.1

    df = sd.diarize_audio(audio)

    assert list(df.columns) == ["start", "end", "speaker"]
    assert len(df) == 3
    # Two unique speakers (segments 0 + 2 are the same speaker).
    assert sorted(df["speaker"].unique().tolist()) == ["SPEAKER_00", "SPEAKER_01"]
    # First-appearance renumbering means segment 0 is SPEAKER_00.
    assert df.iloc[0]["speaker"] == "SPEAKER_00"
    assert df.iloc[2]["speaker"] == "SPEAKER_00"
    assert df.iloc[1]["speaker"] == "SPEAKER_01"


@needs_torch
def test_diarize_audio_returns_empty_dataframe_when_vad_finds_no_speech(
    stub_speechbrain_stack: dict[str, Any],
) -> None:
    stub_speechbrain_stack["vad_segments"] = []
    audio = np.zeros(sd.DEFAULT_SAMPLE_RATE * 2, dtype="float32")

    df = sd.diarize_audio(audio)

    assert list(df.columns) == ["start", "end", "speaker"]
    assert len(df) == 0


def test_diarize_audio_rejects_non_mono_input() -> None:
    stereo = np.zeros((sd.DEFAULT_SAMPLE_RATE * 2, 2), dtype="float32")
    with pytest.raises(sd.SpeechBrainDiarizationError, match="1-D mono"):
        sd.diarize_audio(stereo)


def test_diarize_audio_rejects_non_ndarray_input() -> None:
    with pytest.raises(sd.SpeechBrainDiarizationError, match="numpy.ndarray"):
        sd.diarize_audio([0.0, 0.1, 0.0])  # type: ignore[arg-type]


def test_diarize_audio_returns_empty_for_empty_audio() -> None:
    df = sd.diarize_audio(np.zeros((0,), dtype="float32"))
    assert len(df) == 0


@needs_torch
def test_diarize_audio_passes_savedir_when_cache_dir_configured(
    tmp_path: Path,
    stub_speechbrain_stack: dict[str, Any],
) -> None:
    cache_dir = tmp_path / "speechbrain"
    audio = np.zeros(sd.DEFAULT_SAMPLE_RATE * 6, dtype="float32")
    audio[: sd.DEFAULT_SAMPLE_RATE * 5] = 0.1
    config = sd.DiarizationConfig(cache_dir=cache_dir)

    sd.diarize_audio(audio, config=config)

    assert stub_speechbrain_stack["classifier_savedir"] == str(cache_dir)
    # Cache dir is created even when the test classifier is fake — important
    # so the first real run doesn't race on directory creation.
    assert cache_dir.is_dir()


@needs_torch
def test_diarize_audio_uses_default_cache_dir_when_none_configured(
    monkeypatch: pytest.MonkeyPatch,
    stub_speechbrain_stack: dict[str, Any],
    tmp_path: Path,
) -> None:
    """The shared cache lives at ~/.cache/clipper-tool/speechbrain by
    default so first-run model weights aren't strewn under whatever CWD the
    user happened to be in when they typed `/clip`."""
    fake_default = tmp_path / "default-cache"
    monkeypatch.setattr(sd, "DEFAULT_CACHE_DIR", fake_default)
    audio = np.zeros(sd.DEFAULT_SAMPLE_RATE * 6, dtype="float32")
    audio[: sd.DEFAULT_SAMPLE_RATE * 5] = 0.1

    sd.diarize_audio(audio)

    assert stub_speechbrain_stack["classifier_savedir"] == str(fake_default)
    assert fake_default.is_dir()


@needs_torch
def test_diarize_audio_skips_short_segments_without_length_mismatch(
    stub_speechbrain_stack: dict[str, Any],
) -> None:
    """Regression for a mismatch crash: when silero hands us a short span
    that the embedder rejects, the dropped span must also disappear from
    the DataFrame — otherwise speech_timestamps and speaker_ids end up
    with different lengths and `_build_diarization_dataframe` raises."""
    sample_rate = sd.DEFAULT_SAMPLE_RATE
    # Three VAD spans, but the middle one is shorter than min_speech_duration.
    stub_speechbrain_stack["vad_segments"] = [
        {"start": 0.0, "end": 1.5},
        {"start": 1.6, "end": 1.7},  # 0.1s — below 0.5s threshold
        {"start": 1.9, "end": 3.4},
    ]
    # Only two embeddings will be requested (the middle span is skipped).
    stub_speechbrain_stack["embeddings"] = np.array(
        [
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
        dtype="float32",
    )
    audio = np.zeros(sample_rate * 4, dtype="float32")
    audio[: sample_rate * 4] = 0.1  # non-silent so chunks are extracted

    df = sd.diarize_audio(audio)

    # The skipped span must not appear in the DataFrame, and the row count
    # must match the number of embeddings actually computed.
    assert len(df) == 2
    assert df.iloc[0]["start"] == pytest.approx(0.0)
    # The 0.1s span between 1.6 and 1.7 is gone.
    assert all(row["start"] != pytest.approx(1.6) for _, row in df.iterrows())
    assert df.iloc[1]["start"] == pytest.approx(1.9)


@needs_torch
def test_diarize_audio_returns_empty_when_all_segments_too_short(
    stub_speechbrain_stack: dict[str, Any],
) -> None:
    """If silero hands us only sub-threshold spans, the embedder rejects
    every chunk and we should return an empty DataFrame (the caller then
    falls back to a single fallback speaker) — not crash."""
    stub_speechbrain_stack["vad_segments"] = [
        {"start": 0.0, "end": 0.05},
        {"start": 0.1, "end": 0.15},
    ]
    stub_speechbrain_stack["embeddings"] = np.zeros((0, 3), dtype="float32")
    audio = np.full(sd.DEFAULT_SAMPLE_RATE, 0.1, dtype="float32")

    df = sd.diarize_audio(audio)

    assert list(df.columns) == ["start", "end", "speaker"]
    assert len(df) == 0
