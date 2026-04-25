"""Open-source speaker diarization via SpeechBrain ECAPA-TDNN + silero-VAD.

Drop-in replacement for ``whisperx.DiarizationPipeline`` that produces the same
DataFrame shape (``start``, ``end``, ``speaker``) consumed by
``whisperx.assign_word_speakers``. The point is zero-config diarization: no
HuggingFace token, no license click-through, no env vars. The model files
(``speechbrain/spkrec-ecapa-voxceleb`` weights, silero-VAD JIT script) are
public downloads and cache locally on first use.

Pipeline:

1. **Voice activity detection.** Silero-VAD (MIT, ~2 MB JIT) on the
   16 kHz mono waveform → list of speech timestamps.
2. **Embedding extraction.** SpeechBrain ECAPA-TDNN
   (``speechbrain/spkrec-ecapa-voxceleb``, Apache 2.0 / CC-BY-4.0, ~80 MB)
   → 192-dim embedding per speech segment.
3. **Clustering.** Spectral clustering on the cosine-similarity matrix with
   the eigengap heuristic for k. Tiny-N short-circuits handle the 1- and
   2-segment cases that spectral clustering can't.
4. **DataFrame emit.** ``pandas.DataFrame`` with columns ``start``, ``end``,
   ``speaker`` matching the schema WhisperX's ``assign_word_speakers``
   actually reads (verified against the upstream source).

All third-party imports are lazy so the adapter is importable in
environments without engine extras (tests stub them out via monkeypatch).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import pandas as pd

DEFAULT_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MIN_SPEECH_DURATION_SEC = 0.5
DEFAULT_MAX_EMBED_DURATION_SEC = 30.0
DEFAULT_MAX_SPEAKERS = 8
DEFAULT_TWO_SEGMENT_COSINE_SIMILARITY_THRESHOLD = 0.65
DEFAULT_CACHE_DIR = (
    Path("~/.cache/clippos/speechbrain").expanduser()
)


class SpeechBrainDiarizationError(RuntimeError):
    """Raised when the speechbrain stack is missing or returns invalid output."""


@dataclass(frozen=True)
class DiarizationConfig:
    """User-tunable knobs for the open-source diarization pipeline.

    Defaults are chosen for the clipping use case (10–60 minute videos with
    1–4 speakers). The clustering k upper bound + cosine threshold are
    deliberately conservative to avoid over-segmenting a single speaker into
    multiple labels — a common failure mode that hurts mining signals more
    than under-segmenting does.
    """

    sample_rate: int = DEFAULT_SAMPLE_RATE
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    cache_dir: Path | None = None
    max_speakers: int = DEFAULT_MAX_SPEAKERS
    min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_DURATION_SEC
    max_embed_duration_sec: float = DEFAULT_MAX_EMBED_DURATION_SEC
    two_segment_cosine_threshold: float = (
        DEFAULT_TWO_SEGMENT_COSINE_SIMILARITY_THRESHOLD
    )


def diarize_audio(
    audio: "np.ndarray",
    *,
    config: DiarizationConfig | None = None,
) -> "pd.DataFrame":
    """Run open-source speaker diarization on a 16 kHz mono waveform.

    Returns a DataFrame with columns ``start`` (float seconds), ``end``
    (float seconds), and ``speaker`` (``"SPEAKER_00"``, ``"SPEAKER_01"`` …).
    The shape is compatible with ``whisperx.assign_word_speakers`` — only
    those three columns are read by upstream WhisperX.
    """
    import numpy as np
    import pandas as pd

    cfg = config or DiarizationConfig()
    if not isinstance(audio, np.ndarray):  # type: ignore[arg-type]
        raise SpeechBrainDiarizationError(
            "audio must be a numpy.ndarray of mono PCM samples"
        )
    if audio.ndim != 1:
        raise SpeechBrainDiarizationError(
            f"audio must be 1-D mono samples; got shape {audio.shape}"
        )
    if audio.size == 0:
        return _empty_diarization_dataframe(pd)

    speech_timestamps = _run_vad(audio, cfg)
    if not speech_timestamps:
        return _empty_diarization_dataframe(pd)

    # _extract_embeddings can drop short chunks the embedder rejects, so we
    # round-trip the spans alongside the embedding matrix to keep them in
    # lock-step. Using just `speech_timestamps` here would crash on any
    # audio with mixed-length VAD spans.
    embeddings, kept_spans = _extract_embeddings(audio, speech_timestamps, cfg)
    if not kept_spans:
        return _empty_diarization_dataframe(pd)

    speaker_ids = cluster_embeddings(
        embeddings,
        max_speakers=cfg.max_speakers,
        two_segment_cosine_threshold=cfg.two_segment_cosine_threshold,
    )
    return _build_diarization_dataframe(kept_spans, speaker_ids)


# ---------- VAD ----------


def _run_vad(
    audio: "np.ndarray",
    cfg: DiarizationConfig,
) -> list[dict[str, float]]:
    """Return speech timestamps in seconds via silero-VAD.

    Output shape: ``[{"start": float, "end": float}, ...]``.
    """
    try:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as exc:  # pragma: no cover - exercised at runtime only
        raise SpeechBrainDiarizationError(
            "silero-vad is required for speechbrain diarization. "
            "Install with: pip install silero-vad"
        ) from exc

    model = load_silero_vad()
    waveform = torch.from_numpy(audio.astype("float32", copy=False))
    raw = get_speech_timestamps(
        waveform,
        model,
        sampling_rate=cfg.sample_rate,
        return_seconds=True,
        min_speech_duration_ms=int(cfg.min_speech_duration_sec * 1000),
    )
    cleaned: list[dict[str, float]] = []
    for span in raw or []:
        try:
            start = float(span["start"])
            end = float(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        cleaned.append({"start": start, "end": end})
    cleaned.sort(key=lambda span: span["start"])
    return cleaned


# ---------- embedding extraction ----------


def _extract_embeddings(
    audio: "np.ndarray",
    speech_timestamps: list[dict[str, float]],
    cfg: DiarizationConfig,
) -> tuple["np.ndarray", list[dict[str, float]]]:
    """Run ECAPA-TDNN on each speech segment.

    Returns ``(embeddings, kept_spans)`` paired in the same order — short
    chunks the embedder rejects are dropped from both lists in lock-step,
    so callers can safely iterate them together. The embeddings matrix
    has shape ``(M, 192)`` where ``M = len(kept_spans)`` (M may be < N).
    """
    try:
        import numpy as np
        import torch
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier
    except ImportError as exc:  # pragma: no cover - runtime only
        raise SpeechBrainDiarizationError(
            "speechbrain is required for open-source diarization. "
            "Install with: pip install speechbrain"
        ) from exc

    cache_dir = cfg.cache_dir if cfg.cache_dir is not None else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    classifier = EncoderClassifier.from_hparams(
        source=cfg.embedding_model,
        savedir=str(cache_dir),
    )
    if classifier is None:
        raise SpeechBrainDiarizationError(
            f"failed to load speaker embedder from {cfg.embedding_model}"
        )

    max_samples = int(cfg.max_embed_duration_sec * cfg.sample_rate)
    min_samples = int(cfg.sample_rate * cfg.min_speech_duration_sec)
    embedding_rows: list["np.ndarray"] = []
    kept_spans: list[dict[str, float]] = []
    for span in speech_timestamps:
        start_idx = max(0, int(span["start"] * cfg.sample_rate))
        end_idx = min(audio.shape[0], int(span["end"] * cfg.sample_rate))
        chunk = audio[start_idx:end_idx]
        if chunk.shape[0] < min_samples:
            continue
        if chunk.shape[0] > max_samples:
            chunk = chunk[:max_samples]
        chunk_tensor = torch.from_numpy(
            chunk.astype("float32", copy=False)
        ).unsqueeze(0)
        with torch.no_grad():
            embed = classifier.encode_batch(chunk_tensor)
        vector = embed.detach().squeeze().cpu().numpy()
        if vector.ndim != 1:
            vector = vector.reshape(-1)
        embedding_rows.append(vector)
        kept_spans.append(span)

    if not embedding_rows:
        return np.zeros((0, 0), dtype="float32"), []
    return (
        np.stack(embedding_rows, axis=0).astype("float32", copy=False),
        kept_spans,
    )


# ---------- clustering (pure-numpy, no ML deps) ----------


def cluster_embeddings(
    embeddings: "np.ndarray",
    *,
    max_speakers: int = DEFAULT_MAX_SPEAKERS,
    two_segment_cosine_threshold: float = (
        DEFAULT_TWO_SEGMENT_COSINE_SIMILARITY_THRESHOLD
    ),
) -> list[int]:
    """Group embedding rows into speaker clusters; return per-row labels.

    Tiny-N cases short-circuit because spectral clustering misbehaves below
    three points. For three or more rows we use spectral clustering on the
    cosine-similarity affinity matrix with the eigengap heuristic to pick k.
    """
    import numpy as np

    if embeddings.size == 0:
        return []
    n_segments = embeddings.shape[0]
    if n_segments == 1:
        return [0]
    if n_segments == 2:
        similarity = float(_cosine_similarity_matrix(embeddings)[0, 1])
        return [0, 0] if similarity >= two_segment_cosine_threshold else [0, 1]

    similarity = _cosine_similarity_matrix(embeddings)
    similarity = np.clip(similarity, 0.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    k = _estimate_num_speakers(similarity, max_speakers=max_speakers)
    if k <= 1:
        return [0] * n_segments

    try:
        from sklearn.cluster import SpectralClustering
    except ImportError as exc:  # pragma: no cover
        raise SpeechBrainDiarizationError(
            "scikit-learn is required for spectral clustering. "
            "It is normally installed transitively via whisperx."
        ) from exc

    clusterer = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
    )
    raw_labels = clusterer.fit_predict(similarity)
    return _renumber_labels_by_first_appearance(raw_labels.tolist())


def _estimate_num_speakers(
    similarity: "np.ndarray",
    *,
    max_speakers: int,
) -> int:
    """Eigengap heuristic on the symmetric normalized Laplacian.

    Returns the cluster count that maximizes the gap between consecutive
    eigenvalues, capped at ``max_speakers`` and at the matrix size minus
    one. Falls back to 1 on numerical failure.
    """
    import numpy as np

    n = similarity.shape[0]
    upper = max(1, min(max_speakers, n - 1))
    if upper < 2:
        return 1
    degrees = similarity.sum(axis=1)
    degrees = np.where(degrees > 0, degrees, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(degrees)
    laplacian_norm = (
        np.eye(n) - (d_inv_sqrt[:, None] * similarity) * d_inv_sqrt[None, :]
    )
    laplacian_norm = (laplacian_norm + laplacian_norm.T) / 2  # symmetrize
    try:
        eigenvalues = np.linalg.eigvalsh(laplacian_norm)
    except np.linalg.LinAlgError:
        return 1
    eigenvalues = np.sort(eigenvalues)[: upper + 1]
    if eigenvalues.size < 2:
        return 1
    gaps = np.diff(eigenvalues)
    if gaps.size == 0:
        return 1
    k = int(np.argmax(gaps)) + 1
    return max(1, min(k, upper))


def _cosine_similarity_matrix(matrix: "np.ndarray") -> "np.ndarray":
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    normalized = matrix / norms
    return normalized @ normalized.T


def _renumber_labels_by_first_appearance(labels: list[int]) -> list[int]:
    """Rewrite cluster ids so the first segment is always SPEAKER_00.

    Spectral clustering returns arbitrary integer ids; remapping by first
    appearance gives stable, demo-friendly labels (``SPEAKER_00`` always
    speaks first).
    """
    remap: dict[int, int] = {}
    out: list[int] = []
    for label in labels:
        if label not in remap:
            remap[label] = len(remap)
        out.append(remap[label])
    return out


# ---------- DataFrame emit ----------


def _build_diarization_dataframe(
    speech_timestamps: list[dict[str, float]],
    speaker_ids: list[int],
) -> "pd.DataFrame":
    """Produce a WhisperX-compatible DataFrame.

    Only ``start``, ``end``, and ``speaker`` are read by
    ``whisperx.assign_word_speakers``; we emit exactly those.
    """
    import pandas as pd

    if len(speech_timestamps) != len(speaker_ids):
        raise SpeechBrainDiarizationError(
            "speech_timestamps and speaker_ids must have the same length; "
            f"got {len(speech_timestamps)} vs {len(speaker_ids)}"
        )
    rows: list[dict[str, Any]] = []
    for span, sid in zip(speech_timestamps, speaker_ids, strict=True):
        rows.append(
            {
                "start": float(span["start"]),
                "end": float(span["end"]),
                "speaker": _format_speaker_label(int(sid)),
            }
        )
    return pd.DataFrame(rows, columns=["start", "end", "speaker"])


def _empty_diarization_dataframe(pd_module: Any) -> "pd.DataFrame":
    """Return an empty DataFrame with the expected schema.

    Returning an empty DataFrame (rather than raising) is the right behavior
    for silent or VAD-rejected audio: ``assign_word_speakers`` short-circuits
    to "no diarization" and every transcript segment falls back to the
    default speaker label, which the normalizer turns into ``"unknown"``.
    """
    return pd_module.DataFrame(columns=["start", "end", "speaker"])


def _format_speaker_label(speaker_id: int) -> str:
    """``SPEAKER_00``, ``SPEAKER_01`` … matches pyannote's convention."""
    return f"SPEAKER_{speaker_id:02d}"
