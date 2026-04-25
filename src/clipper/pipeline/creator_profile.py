"""Creator-profile learning loop.

Phase A captures per-clip feedback (kept vs. skipped + free-text notes) and
appends it to a global ``history.jsonl``. Phase B aggregates the history into
patterns (length bias, spike-category bias, score disagreement, ratio bias)
with a confidence score and a suggested memory string the harness can save.

The module is self-contained and pure-functional: load/aggregate/detect/format
take data, return data. Persistence helpers live at the bottom. The skill
scripts drive this module, they do not contain its logic.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

HISTORY_VERSION = 1


@dataclass(frozen=True)
class HistoryEntry:
    """One outcome — did this specific clip get posted or skipped."""

    job_id: str
    clip_id: str
    recorded_at: str
    duration_seconds: float
    score: float
    spike_categories: tuple[str, ...]
    ratios: tuple[str, ...]
    title: str
    posted: bool
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["spike_categories"] = list(self.spike_categories)
        payload["ratios"] = list(self.ratios)
        payload["history_version"] = HISTORY_VERSION
        return payload

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "HistoryEntry":
        return cls(
            job_id=str(data["job_id"]),
            clip_id=str(data["clip_id"]),
            recorded_at=str(data["recorded_at"]),
            duration_seconds=float(data["duration_seconds"]),
            score=float(data["score"]),
            spike_categories=tuple(data.get("spike_categories") or []),
            ratios=tuple(data.get("ratios") or []),
            title=str(data.get("title", "")),
            posted=bool(data.get("posted", False)),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class Pattern:
    """A detected regularity in how the user treats a class of clips."""

    kind: str  # "length" | "spike" | "score" | "ratio"
    rule: str
    confidence: str  # "low" | "medium" | "high"
    sample_size: int
    keep_rate: float
    overall_keep_rate: float
    suggested_memory: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# ---------- persistence ----------


def append_history(path: Path, entries: Iterable[HistoryEntry]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("a", encoding="utf-8") as sink:
        for entry in entries:
            sink.write(json.dumps(entry.to_json()) + "\n")
            written += 1
    return written


def load_history(path: Path) -> list[HistoryEntry]:
    if not path.exists():
        return []
    entries: list[HistoryEntry] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            entries.append(HistoryEntry.from_json(payload))
        except (KeyError, ValueError, TypeError):
            continue
    return entries


def latest_entries_by_clip(entries: list[HistoryEntry]) -> list[HistoryEntry]:
    """Keep only the most recent entry per (job_id, clip_id) pair.

    Feedback may be resubmitted if the user changes their mind; the later
    row should win for pattern detection so the skill learns from their
    latest decision rather than re-counting both.
    """
    latest: dict[tuple[str, str], HistoryEntry] = {}
    for entry in entries:
        key = (entry.job_id, entry.clip_id)
        latest[key] = entry  # later rows overwrite earlier ones
    return list(latest.values())


# ---------- aggregation ----------


def summarize(entries: list[HistoryEntry]) -> dict[str, Any]:
    total = len(entries)
    kept = sum(1 for e in entries if e.posted)
    skipped = total - kept
    keep_rate = _safe_rate(kept, total)
    by_ratio: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "kept": 0}
    )
    by_spike: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "kept": 0}
    )
    for entry in entries:
        for ratio in entry.ratios or ():
            bucket = by_ratio[ratio]
            bucket["total"] += 1
            bucket["kept"] += 1 if entry.posted else 0
        for spike in entry.spike_categories or ():
            bucket = by_spike[spike]
            bucket["total"] += 1
            bucket["kept"] += 1 if entry.posted else 0
    return {
        "total_clips": total,
        "kept": kept,
        "skipped": skipped,
        "keep_rate": keep_rate,
        "per_ratio": {
            ratio: {
                **counts,
                "keep_rate": _safe_rate(counts["kept"], counts["total"]),
            }
            for ratio, counts in sorted(by_ratio.items())
        },
        "per_spike_category": {
            spike: {
                **counts,
                "keep_rate": _safe_rate(counts["kept"], counts["total"]),
            }
            for spike, counts in sorted(by_spike.items())
        },
    }


# ---------- pattern detection (Phase B) ----------


LENGTH_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("under_20s", 0.0, 20.0),
    ("20_to_30s", 20.0, 30.0),
    ("30_to_45s", 30.0, 45.0),
    ("45_to_60s", 45.0, 60.0),
    ("over_60s", 60.0, float("inf")),
)

_MIN_HIGH = 15
_MIN_MEDIUM = 8
_MIN_LOW = 5
_DEV_HIGH = 0.30
_DEV_MEDIUM = 0.25
_DEV_LOW = 0.20


def detect_patterns(entries: list[HistoryEntry]) -> list[Pattern]:
    if len(entries) < _MIN_LOW:
        return []
    overall_keep_rate = _safe_rate(
        sum(1 for e in entries if e.posted), len(entries)
    )
    patterns: list[Pattern] = []
    patterns.extend(_length_patterns(entries, overall_keep_rate))
    patterns.extend(_spike_patterns(entries, overall_keep_rate))
    patterns.extend(_score_patterns(entries, overall_keep_rate))
    patterns.extend(_ratio_patterns(entries, overall_keep_rate))
    return sorted(
        patterns,
        key=lambda p: (_CONFIDENCE_RANK[p.confidence], -p.sample_size),
        reverse=True,
    )


_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def _length_patterns(
    entries: list[HistoryEntry], overall_keep_rate: float
) -> list[Pattern]:
    patterns: list[Pattern] = []
    for label, low, high in LENGTH_BUCKETS:
        bucket = [
            e for e in entries if low <= e.duration_seconds < high
        ]
        confidence = _confidence(len(bucket), bucket, overall_keep_rate)
        if confidence is None:
            continue
        keep_rate = _safe_rate(sum(1 for e in bucket if e.posted), len(bucket))
        direction = "keeps" if keep_rate > overall_keep_rate else "skips"
        human_label = _length_label(label)
        rule = (
            f"User {direction} clips {human_label} "
            f"({int(keep_rate * 100)}% keep rate vs. "
            f"{int(overall_keep_rate * 100)}% overall)."
        )
        suggested_memory = _length_memory(direction, human_label)
        patterns.append(
            Pattern(
                kind="length",
                rule=rule,
                confidence=confidence,
                sample_size=len(bucket),
                keep_rate=keep_rate,
                overall_keep_rate=overall_keep_rate,
                suggested_memory=suggested_memory,
                evidence={"bucket": label, "range": [low, high]},
            )
        )
    return patterns


def _spike_patterns(
    entries: list[HistoryEntry], overall_keep_rate: float
) -> list[Pattern]:
    counts: dict[str, list[HistoryEntry]] = defaultdict(list)
    for entry in entries:
        for spike in entry.spike_categories or ():
            counts[spike].append(entry)
    patterns: list[Pattern] = []
    for spike, bucket in sorted(counts.items()):
        confidence = _confidence(len(bucket), bucket, overall_keep_rate)
        if confidence is None:
            continue
        keep_rate = _safe_rate(sum(1 for e in bucket if e.posted), len(bucket))
        direction = "keeps" if keep_rate > overall_keep_rate else "skips"
        rule = (
            f"User {direction} '{spike}' clips "
            f"({int(keep_rate * 100)}% keep rate vs. "
            f"{int(overall_keep_rate * 100)}% overall)."
        )
        suggested_memory = (
            f"Clip creator {direction} '{spike}' spike clips; "
            "weight that signal accordingly when scoring."
        )
        patterns.append(
            Pattern(
                kind="spike",
                rule=rule,
                confidence=confidence,
                sample_size=len(bucket),
                keep_rate=keep_rate,
                overall_keep_rate=overall_keep_rate,
                suggested_memory=suggested_memory,
                evidence={"spike_category": spike},
            )
        )
    return patterns


def _score_patterns(
    entries: list[HistoryEntry], overall_keep_rate: float
) -> list[Pattern]:
    high_score_bucket = [e for e in entries if e.score >= 0.85]
    if len(high_score_bucket) < _MIN_MEDIUM:
        return []
    keep_rate = _safe_rate(
        sum(1 for e in high_score_bucket if e.posted),
        len(high_score_bucket),
    )
    deviation = overall_keep_rate - keep_rate
    if deviation < _DEV_MEDIUM:
        return []
    confidence = (
        "high"
        if len(high_score_bucket) >= _MIN_HIGH and deviation >= _DEV_HIGH
        else "medium"
    )
    rule = (
        "User skips many high-scoring clips "
        f"({int(keep_rate * 100)}% keep rate at score >= 0.85 vs. "
        f"{int(overall_keep_rate * 100)}% overall)."
    )
    suggested_memory = (
        "Rubric score alone is not a reliable signal for this creator; "
        "weight spike categories, pacing, and concrete payoff over raw score."
    )
    return [
        Pattern(
            kind="score",
            rule=rule,
            confidence=confidence,
            sample_size=len(high_score_bucket),
            keep_rate=keep_rate,
            overall_keep_rate=overall_keep_rate,
            suggested_memory=suggested_memory,
            evidence={"threshold": 0.85},
        )
    ]


def _ratio_patterns(
    entries: list[HistoryEntry], overall_keep_rate: float
) -> list[Pattern]:
    per_ratio: dict[str, list[HistoryEntry]] = defaultdict(list)
    for entry in entries:
        for ratio in entry.ratios or ():
            per_ratio[ratio].append(entry)
    patterns: list[Pattern] = []
    for ratio, bucket in sorted(per_ratio.items()):
        confidence = _confidence(len(bucket), bucket, overall_keep_rate)
        if confidence is None:
            continue
        keep_rate = _safe_rate(sum(1 for e in bucket if e.posted), len(bucket))
        direction = "keeps" if keep_rate > overall_keep_rate else "skips"
        rule = (
            f"User {direction} {ratio} renders "
            f"({int(keep_rate * 100)}% keep rate vs. "
            f"{int(overall_keep_rate * 100)}% overall)."
        )
        suggested_memory = (
            f"Clip creator mostly {direction} the {ratio} ratio; "
            "consider adjusting default ratios accordingly."
        )
        patterns.append(
            Pattern(
                kind="ratio",
                rule=rule,
                confidence=confidence,
                sample_size=len(bucket),
                keep_rate=keep_rate,
                overall_keep_rate=overall_keep_rate,
                suggested_memory=suggested_memory,
                evidence={"ratio": ratio},
            )
        )
    return patterns


# ---------- helpers ----------


def _confidence(
    sample_size: int,
    bucket: list[HistoryEntry],
    overall_keep_rate: float,
) -> str | None:
    if sample_size < _MIN_LOW:
        return None
    keep_rate = _safe_rate(sum(1 for e in bucket if e.posted), sample_size)
    deviation = abs(keep_rate - overall_keep_rate)
    if sample_size >= _MIN_HIGH and deviation >= _DEV_HIGH:
        return "high"
    if sample_size >= _MIN_MEDIUM and deviation >= _DEV_MEDIUM:
        return "medium"
    if sample_size >= _MIN_LOW and deviation >= _DEV_LOW:
        return "low"
    return None


def _length_label(bucket_key: str) -> str:
    mapping = {
        "under_20s": "under 20s",
        "20_to_30s": "between 20 and 30s",
        "30_to_45s": "between 30 and 45s",
        "45_to_60s": "between 45 and 60s",
        "over_60s": "over 60s",
    }
    return mapping.get(bucket_key, bucket_key)


def _length_memory(direction: str, human_label: str) -> str:
    return (
        f"Clip creator {direction} clips {human_label}; "
        "bias candidate scoring accordingly."
    )


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
