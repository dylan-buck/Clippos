from __future__ import annotations

from clipper.models.render import CaptionLine, CaptionWord
from clipper.pipeline.transcribe import TranscriptTimeline, TranscriptWord

DEFAULT_MAX_WORDS_PER_LINE = 5
DEFAULT_MAX_LINE_DURATION = 1.2
DEFAULT_MAX_LINE_GAP = 0.35
MIN_EMPHASIS_LENGTH = 6

STOPWORDS = frozenset(
    {
        "about",
        "actually",
        "because",
        "before",
        "between",
        "really",
        "should",
        "something",
        "through",
        "together",
    }
)


def build_caption_plan(
    transcript: TranscriptTimeline,
    *,
    start_seconds: float,
    end_seconds: float,
    max_words_per_line: int = DEFAULT_MAX_WORDS_PER_LINE,
    max_line_duration: float = DEFAULT_MAX_LINE_DURATION,
    max_line_gap: float = DEFAULT_MAX_LINE_GAP,
) -> list[CaptionLine]:
    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds")
    if max_words_per_line <= 0:
        raise ValueError("max_words_per_line must be positive")
    if max_line_duration <= 0:
        raise ValueError("max_line_duration must be positive")

    caption_words = _collect_caption_words(
        transcript, start_seconds=start_seconds, end_seconds=end_seconds
    )
    if not caption_words:
        return []

    return _group_into_lines(
        caption_words,
        max_words_per_line=max_words_per_line,
        max_line_duration=max_line_duration,
        max_line_gap=max_line_gap,
    )


def _collect_caption_words(
    transcript: TranscriptTimeline,
    *,
    start_seconds: float,
    end_seconds: float,
) -> list[CaptionWord]:
    collected: list[CaptionWord] = []
    for segment in transcript.segments:
        if segment.end_seconds <= start_seconds or segment.start_seconds >= end_seconds:
            continue
        for word in segment.words:
            if word.end_seconds <= start_seconds or word.start_seconds >= end_seconds:
                continue
            word_start = max(word.start_seconds, start_seconds) - start_seconds
            word_end = min(word.end_seconds, end_seconds) - start_seconds
            if word_end <= word_start:
                continue
            collected.append(
                CaptionWord(
                    text=word.text,
                    start_seconds=word_start,
                    end_seconds=word_end,
                    emphasis=_should_emphasize(word),
                )
            )
    collected.sort(key=lambda word: word.start_seconds)
    return collected


def _should_emphasize(word: TranscriptWord) -> bool:
    token = word.text.strip()
    if not token:
        return False
    stripped = token.strip(".,!?;:\"'()[]{}")
    if not stripped:
        return False
    if any(character.isdigit() for character in stripped):
        return True
    if stripped.isupper() and len(stripped) > 1:
        return True
    lowered = stripped.lower()
    if lowered in STOPWORDS:
        return False
    return len(stripped) >= MIN_EMPHASIS_LENGTH


def _group_into_lines(
    words: list[CaptionWord],
    *,
    max_words_per_line: int,
    max_line_duration: float,
    max_line_gap: float,
) -> list[CaptionLine]:
    lines: list[CaptionLine] = []
    current: list[CaptionWord] = []

    for word in words:
        if not current:
            current.append(word)
            continue
        line_start = current[0].start_seconds
        line_gap = word.start_seconds - current[-1].end_seconds
        line_duration = word.end_seconds - line_start
        if (
            len(current) >= max_words_per_line
            or line_duration > max_line_duration
            or line_gap > max_line_gap
        ):
            lines.append(_finalize_line(current))
            current = [word]
        else:
            current.append(word)

    if current:
        lines.append(_finalize_line(current))

    return lines


def _finalize_line(words: list[CaptionWord]) -> CaptionLine:
    return CaptionLine(
        start_seconds=words[0].start_seconds,
        end_seconds=words[-1].end_seconds,
        text=" ".join(word.text for word in words),
        words=list(words),
    )
