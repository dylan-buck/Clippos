from __future__ import annotations

from pathlib import Path

from clippos.pipeline import creator_profile as cp


def _entry(
    *,
    clip_id: str = "c1",
    posted: bool = False,
    duration: float = 30.0,
    score: float = 0.8,
    spike_categories: tuple[str, ...] = (),
    ratios: tuple[str, ...] = ("9:16",),
    notes: str = "",
    job_id: str = "job-a",
) -> cp.HistoryEntry:
    return cp.HistoryEntry(
        job_id=job_id,
        clip_id=clip_id,
        recorded_at="2026-04-24T00:00:00+00:00",
        duration_seconds=duration,
        score=score,
        spike_categories=spike_categories,
        ratios=ratios,
        title=f"Title {clip_id}",
        posted=posted,
        notes=notes,
    )


def test_roundtrip_history_entry() -> None:
    original = _entry(
        clip_id="c7",
        posted=True,
        spike_categories=("taboo", "absurdity"),
        ratios=("9:16", "1:1"),
        notes="posted to TikTok",
    )

    roundtripped = cp.HistoryEntry.from_json(original.to_json())
    assert roundtripped == original


def test_append_and_load_history_preserves_order(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    batch_one = [_entry(clip_id="c1"), _entry(clip_id="c2", posted=True)]
    batch_two = [_entry(clip_id="c3", posted=True, job_id="job-b")]

    assert cp.append_history(path, batch_one) == 2
    assert cp.append_history(path, batch_two) == 1

    entries = cp.load_history(path)
    assert [entry.clip_id for entry in entries] == ["c1", "c2", "c3"]
    assert entries[2].job_id == "job-b"


def test_latest_entries_by_clip_keeps_last_feedback_per_pair() -> None:
    initial = _entry(clip_id="c1", posted=False, notes="first pass")
    correction = _entry(clip_id="c1", posted=True, notes="changed my mind")
    other = _entry(clip_id="c2", posted=True)

    latest = cp.latest_entries_by_clip([initial, correction, other])
    by_clip = {entry.clip_id: entry for entry in latest}

    assert len(latest) == 2
    assert by_clip["c1"].posted is True
    assert by_clip["c1"].notes == "changed my mind"
    assert by_clip["c2"].posted is True


def test_load_history_skips_blank_and_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    path.write_text(
        "\n"
        '{"not": "valid"}\n'
        "not-json-at-all\n"
        '{"job_id": "j", "clip_id": "c", "recorded_at": "t", '
        '"duration_seconds": 10.0, "score": 0.5, "spike_categories": [], '
        '"ratios": [], "title": "", "posted": true}\n',
        encoding="utf-8",
    )

    entries = cp.load_history(path)
    assert len(entries) == 1
    assert entries[0].clip_id == "c"


def test_summarize_counts_overall_and_per_bucket() -> None:
    entries = [
        _entry(clip_id="c1", posted=True, ratios=("9:16",), spike_categories=("taboo",)),
        _entry(clip_id="c2", posted=False, ratios=("9:16", "1:1"), spike_categories=("taboo",)),
        _entry(clip_id="c3", posted=True, ratios=("1:1",), spike_categories=("absurdity",)),
    ]

    result = cp.summarize(entries)
    assert result["total_clips"] == 3
    assert result["kept"] == 2
    assert result["skipped"] == 1
    assert result["keep_rate"] == round(2 / 3, 4)
    assert result["per_ratio"]["9:16"]["total"] == 2
    assert result["per_ratio"]["9:16"]["kept"] == 1
    assert result["per_spike_category"]["taboo"]["keep_rate"] == 0.5


def test_detect_patterns_returns_empty_when_history_is_small() -> None:
    assert cp.detect_patterns([_entry() for _ in range(3)]) == []


def test_detect_patterns_flags_length_bias_over_sixty_seconds() -> None:
    short_entries = [
        _entry(clip_id=f"s{i}", duration=25.0, posted=True)
        for i in range(10)
    ]
    long_entries = [
        _entry(clip_id=f"l{i}", duration=75.0, posted=False)
        for i in range(8)
    ]

    patterns = cp.detect_patterns(short_entries + long_entries)
    long_patterns = [p for p in patterns if p.evidence.get("bucket") == "over_60s"]

    assert long_patterns, "expected a length bias pattern on the over-60s bucket"
    pattern = long_patterns[0]
    assert pattern.keep_rate == 0.0
    assert pattern.sample_size == 8
    assert pattern.confidence in {"medium", "high"}
    assert "over 60s" in pattern.suggested_memory


def test_detect_patterns_flags_spike_category_bias() -> None:
    entries = []
    # User keeps 'taboo' at high rate
    entries.extend(
        _entry(
            clip_id=f"t{i}",
            posted=True,
            spike_categories=("taboo",),
        )
        for i in range(10)
    )
    # User skips 'rambling' coded as spike at low rate
    entries.extend(
        _entry(
            clip_id=f"a{i}",
            posted=False,
            spike_categories=("absurdity",),
        )
        for i in range(10)
    )

    patterns = cp.detect_patterns(entries)
    spike_patterns = [p for p in patterns if p.kind == "spike"]
    assert any(
        p.evidence["spike_category"] == "taboo" and p.keep_rate > 0.5
        for p in spike_patterns
    )
    assert any(
        p.evidence["spike_category"] == "absurdity" and p.keep_rate < 0.5
        for p in spike_patterns
    )


def test_detect_patterns_flags_score_disagreement_when_user_skips_high_scoring_clips() -> None:
    entries = [
        _entry(clip_id=f"h{i}", score=0.9, posted=False) for i in range(10)
    ]
    entries.extend(
        _entry(clip_id=f"l{i}", score=0.5, posted=True) for i in range(10)
    )

    patterns = cp.detect_patterns(entries)
    score_patterns = [p for p in patterns if p.kind == "score"]
    assert score_patterns
    pattern = score_patterns[0]
    assert pattern.sample_size == 10
    assert pattern.keep_rate == 0.0
    assert "Rubric score alone" in pattern.suggested_memory


def test_detect_patterns_sorted_with_high_confidence_first() -> None:
    entries = []
    entries.extend(
        _entry(clip_id=f"l{i}", duration=80.0, posted=False) for i in range(20)
    )
    entries.extend(
        _entry(clip_id=f"s{i}", duration=20.0, posted=True) for i in range(20)
    )

    patterns = cp.detect_patterns(entries)
    assert patterns[0].confidence == "high"
