import pytest

from clippos.pipeline.captions import build_caption_plan
from clippos.pipeline.transcribe import (
    TranscriptSegment,
    TranscriptTimeline,
    TranscriptWord,
)


def _word(text: str, start: float, end: float) -> TranscriptWord:
    return TranscriptWord(
        text=text, start_seconds=start, end_seconds=end, confidence=0.99
    )


def _segment(speaker: str, words: list[TranscriptWord]) -> TranscriptSegment:
    return TranscriptSegment(
        speaker=speaker,
        start_seconds=words[0].start_seconds,
        end_seconds=words[-1].end_seconds,
        text=" ".join(word.text for word in words),
        words=words,
    )


def test_build_caption_plan_retimes_words_relative_to_clip_start() -> None:
    transcript = TranscriptTimeline(
        segments=[
            _segment(
                "speaker_1",
                [
                    _word("Look", 10.0, 10.3),
                    _word("secret", 10.3, 10.9),
                ],
            )
        ]
    )

    lines = build_caption_plan(transcript, start_seconds=10.0, end_seconds=11.0)

    assert len(lines) == 1
    assert lines[0].words[0].start_seconds == pytest.approx(0.0)
    assert lines[0].words[0].end_seconds == pytest.approx(0.3)
    assert lines[0].words[1].start_seconds == pytest.approx(0.3)


def test_build_caption_plan_emphasizes_long_content_words() -> None:
    transcript = TranscriptTimeline(
        segments=[
            _segment(
                "speaker_1",
                [
                    _word("Look", 0.0, 0.2),
                    _word("at", 0.21, 0.3),
                    _word("secret", 0.31, 0.8),
                    _word("tradeoffs", 0.81, 1.3),
                ],
            )
        ]
    )

    lines = build_caption_plan(transcript, start_seconds=0.0, end_seconds=1.5)

    emphasis_by_text = {
        word.text: word.emphasis for line in lines for word in line.words
    }
    assert emphasis_by_text == {
        "Look": False,
        "at": False,
        "secret": True,
        "tradeoffs": True,
    }


def test_build_caption_plan_emphasizes_numbers_and_caps() -> None:
    transcript = TranscriptTimeline(
        segments=[
            _segment(
                "speaker_1",
                [
                    _word("STOP", 0.0, 0.2),
                    _word("in", 0.21, 0.3),
                    _word("2024", 0.31, 0.8),
                ],
            )
        ]
    )

    lines = build_caption_plan(transcript, start_seconds=0.0, end_seconds=1.0)

    assert len(lines) == 1
    emphasis_by_text = {word.text: word.emphasis for word in lines[0].words}
    assert emphasis_by_text == {"STOP": True, "in": False, "2024": True}


def test_build_caption_plan_groups_into_short_lines() -> None:
    words = [_word(f"word{i}", i * 0.2, i * 0.2 + 0.18) for i in range(12)]
    transcript = TranscriptTimeline(segments=[_segment("speaker_1", words)])

    lines = build_caption_plan(
        transcript,
        start_seconds=0.0,
        end_seconds=3.0,
        max_words_per_line=4,
    )

    assert [len(line.words) for line in lines] == [4, 4, 4]
    for line in lines:
        assert line.end_seconds > line.start_seconds


def test_build_caption_plan_breaks_line_on_silence_gap() -> None:
    words = [
        _word("hello", 0.0, 0.3),
        _word("world", 0.35, 0.6),
        _word("later", 2.0, 2.4),
    ]
    transcript = TranscriptTimeline(segments=[_segment("speaker_1", words)])

    lines = build_caption_plan(transcript, start_seconds=0.0, end_seconds=3.0)

    assert len(lines) == 2
    assert [word.text for word in lines[0].words] == ["hello", "world"]
    assert [word.text for word in lines[1].words] == ["later"]


def test_build_caption_plan_clips_words_spanning_boundary() -> None:
    transcript = TranscriptTimeline(
        segments=[
            _segment(
                "speaker_1",
                [
                    _word("edge", 0.8, 1.5),
                    _word("after", 2.0, 2.4),
                ],
            )
        ]
    )

    lines = build_caption_plan(transcript, start_seconds=1.0, end_seconds=2.2)

    flat = [word for line in lines for word in line.words]
    assert [word.text for word in flat] == ["edge", "after"]
    assert flat[0].start_seconds == pytest.approx(0.0)
    assert flat[0].end_seconds == pytest.approx(0.5)
    assert flat[1].start_seconds == pytest.approx(1.0)
    assert flat[1].end_seconds == pytest.approx(1.2)


def test_build_caption_plan_returns_empty_for_segment_without_overlap() -> None:
    transcript = TranscriptTimeline(
        segments=[_segment("speaker_1", [_word("out", 5.0, 5.5)])]
    )

    assert build_caption_plan(transcript, start_seconds=0.0, end_seconds=1.0) == []


def test_build_caption_plan_rejects_invalid_bounds() -> None:
    transcript = TranscriptTimeline(segments=[])
    with pytest.raises(ValueError):
        build_caption_plan(transcript, start_seconds=1.0, end_seconds=1.0)
