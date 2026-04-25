from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from clipper.models.candidate import CandidateClip
from clipper.pipeline.transcribe import (
    TranscriptSegment,
    TranscriptTimeline,
    TranscriptWord,
)
from clipper.pipeline.vision import VisionFrame, VisionTimeline

# Keyword buckets map roughly onto the spike categories from the design doc.
CONTROVERSY_KEYWORDS = frozenset(
    {
        "ban",
        "banned",
        "illegal",
        "scam",
        "lawsuit",
        "sued",
        "fight",
        "versus",
        "controversial",
    }
)
TABOO_KEYWORDS = frozenset(
    {"taboo", "secret", "hidden", "forbidden", "classified", "confidential"}
)
ABSURDITY_KEYWORDS = frozenset(
    {"crazy", "wild", "insane", "absurd", "ridiculous", "unreal", "bonkers"}
)
ACTION_KEYWORDS = frozenset(
    {
        "exploded",
        "smashed",
        "crashed",
        "jumped",
        "snapped",
        "ran",
        "rushed",
        "flipped",
    }
)
CONFRONTATION_KEYWORDS = frozenset(
    {
        "angry",
        "furious",
        "screamed",
        "yelled",
        "argued",
        "interrupted",
        "snapped",
        "hate",
    }
)
USEFUL_CLAIM_KEYWORDS = frozenset(
    {
        "learn",
        "lesson",
        "tip",
        "trick",
        "truth",
        "rule",
        "insight",
        "framework",
        "pattern",
    }
)
PAYOFF_KEYWORDS = frozenset(
    {
        "payoff",
        "reveal",
        "outcome",
        "result",
        "aftermath",
        "turns",
        "finally",
        "boom",
    }
)
PAYOFF_PHRASES = (
    "turns out",
    "the result was",
    "so it turns out",
    "and that's when",
    "what actually happened",
    "here's the kicker",
)
INTERJECTION_TOKENS = frozenset(
    {
        "wait",
        "whoa",
        "woah",
        "listen",
        "honestly",
        "seriously",
        "actually",
        "literally",
    }
)
HOOK_PHRASES = (
    "here's the thing",
    "let me tell you",
    "you won't believe",
    "the crazy part is",
    "nobody talks about",
    "this is wild",
)
HOOK_TOKENS = frozenset({"wait", "listen", "okay", "right", "so", "imagine", "picture"})
BURIED_LEAD_PHRASES = (
    "only makes sense if",
    "if you watched",
    "if you saw",
    "as i said earlier",
    "in the last part",
    "like i mentioned",
    "earlier we covered",
)

_WORD_RE = re.compile(r"\b[\w']+\b")
_NUMERIC_RE = re.compile(r"\b\d+(?:[\.,]\d+)?\b|\b\d+(?:st|nd|rd|th)\b")


@dataclass(frozen=True)
class DurationPolicy:
    min_seconds: float = 15.0
    max_seconds: float = 45.0

    def __post_init__(self) -> None:
        if self.min_seconds <= 0 or self.max_seconds <= 0:
            raise ValueError("duration bounds must be positive")
        if self.max_seconds < self.min_seconds:
            raise ValueError("max_seconds must be >= min_seconds")


@dataclass(frozen=True)
class ScoringWeights:
    base: float = 0.10
    hook: float = 0.12
    keyword: float = 0.12
    numeric: float = 0.05
    interjection: float = 0.05
    payoff: float = 0.12
    question_to_answer: float = 0.08
    motion: float = 0.08
    shot_change: float = 0.05
    face_presence: float = 0.05
    speaker_interaction: float = 0.05
    delivery_variance: float = 0.03


@dataclass(frozen=True)
class Penalties:
    buried_lead: float = 0.18
    dangling_question: float = 0.08
    rambling_middle: float = 0.12


@dataclass(frozen=True)
class MiningConfig:
    duration_policy: DurationPolicy = field(default_factory=DurationPolicy)
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    penalties: Penalties = field(default_factory=Penalties)
    score_floor: float = 0.35
    min_candidates: int = 5
    max_overlap_ratio: float = 0.5
    rambling_motion_ceiling: float = 0.25
    rambling_keyword_floor: float = 0.05


@dataclass(frozen=True)
class WindowSignals:
    hook: float
    keyword: float
    numeric: float
    interjection: float
    payoff: float
    question_to_answer: float
    motion: float
    shot_change: float
    face_presence: float
    speaker_interaction: float
    delivery_variance: float
    buried_lead: bool
    dangling_question: bool
    rambling_middle: bool


@dataclass(frozen=True)
class ScoredWindow:
    segments: tuple[TranscriptSegment, ...]
    signals: WindowSignals
    score: float

    @property
    def start_seconds(self) -> float:
        return self.segments[0].start_seconds

    @property
    def end_seconds(self) -> float:
        return self.segments[-1].end_seconds


def generate_candidates(
    transcript_timeline: TranscriptTimeline,
    vision_timeline: VisionTimeline,
    max_candidates: int = 12,
    *,
    config: MiningConfig | None = None,
) -> list[CandidateClip]:
    windows = mine_windows(
        transcript_timeline,
        vision_timeline,
        max_candidates,
        config=config,
    )
    return [
        to_candidate_clip(rank=index, scored=sw) for index, sw in enumerate(windows)
    ]


def mine_windows(
    transcript_timeline: TranscriptTimeline,
    vision_timeline: VisionTimeline,
    max_candidates: int = 12,
    *,
    config: MiningConfig | None = None,
) -> list[ScoredWindow]:
    cfg = config or MiningConfig()
    if max_candidates <= 0:
        return []

    windows = enumerate_segment_windows(
        transcript_timeline.segments, cfg.duration_policy
    )
    scored: list[ScoredWindow] = [
        _score_window(window, vision_timeline.frames, cfg) for window in windows
    ]
    scored.sort(key=lambda sw: sw.score, reverse=True)
    above_floor = [sw for sw in scored if sw.score >= cfg.score_floor]
    selected = _greedy_deduplicate(
        above_floor, max_overlap_ratio=cfg.max_overlap_ratio
    )
    minimum = min(max_candidates, max(cfg.min_candidates, 0))
    if len(selected) < minimum:
        selected = _fill_minimum_candidates(
            selected,
            scored,
            minimum=minimum,
            max_overlap_ratio=cfg.max_overlap_ratio,
        )
    return selected[:max_candidates]


def to_candidate_clip(*, rank: int, scored: ScoredWindow) -> CandidateClip:
    return CandidateClip(
        clip_id=f"clip-{rank:03d}",
        start_seconds=scored.start_seconds,
        end_seconds=scored.end_seconds,
        score=scored.score,
        reasons=derive_reasons(scored.signals),
        spike_categories=derive_spike_categories(scored.signals),
    )


def enumerate_segment_windows(
    segments: list[TranscriptSegment], policy: DurationPolicy
) -> list[tuple[TranscriptSegment, ...]]:
    windows: list[tuple[TranscriptSegment, ...]] = []
    for start_index in range(len(segments)):
        for end_index in range(start_index + 1, len(segments) + 1):
            subset = segments[start_index:end_index]
            duration = subset[-1].end_seconds - subset[0].start_seconds
            if duration > policy.max_seconds:
                break
            if duration >= policy.min_seconds:
                windows.append(tuple(subset))
    return windows


def score_hook_strength(first_segment_text: str) -> float:
    lowered = first_segment_text.lower()
    phrase_hits = sum(1 for phrase in HOOK_PHRASES if phrase in lowered)
    tokens = _tokens(lowered)
    first_four = tokens[:4]
    token_hits = sum(1 for token in first_four if token in HOOK_TOKENS)
    keyword_bonus = score_keyword_spike(first_segment_text)
    return _clamp01(
        0.35 * min(phrase_hits, 2) + 0.2 * min(token_hits, 2) + 0.5 * keyword_bonus
    )


def score_keyword_spike(text: str) -> float:
    tokens = set(_tokens(text))
    if not tokens:
        return 0.0
    categories = (
        CONTROVERSY_KEYWORDS,
        TABOO_KEYWORDS,
        ABSURDITY_KEYWORDS,
        ACTION_KEYWORDS,
        CONFRONTATION_KEYWORDS,
        USEFUL_CLAIM_KEYWORDS,
    )
    category_hits = sum(1 for bucket in categories if tokens & bucket)
    return _clamp01(category_hits / len(categories))


def score_numeric_density(text: str) -> float:
    tokens = _tokens(text)
    if not tokens:
        return 0.0
    numeric_hits = len(_NUMERIC_RE.findall(text))
    density = numeric_hits / len(tokens)
    return _clamp01(density * 5)


def score_interjection_density(text: str) -> float:
    tokens = _tokens(text)
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in INTERJECTION_TOKENS)
    return _clamp01(hits / max(len(tokens) / 20, 1))


def score_payoff_signal(text: str) -> float:
    lowered = text.lower()
    phrase_hits = sum(1 for phrase in PAYOFF_PHRASES if phrase in lowered)
    tokens = set(_tokens(lowered))
    keyword_hits = len(tokens & PAYOFF_KEYWORDS)
    if phrase_hits == 0 and keyword_hits == 0:
        return 0.0
    return _clamp01(0.6 * min(phrase_hits, 2) + 0.25 * min(keyword_hits, 3))


def score_question_to_answer(segments: tuple[TranscriptSegment, ...]) -> float:
    if len(segments) < 2:
        return 0.0
    midpoint = max(len(segments) // 2, 1)
    first_half_text = " ".join(segment.text for segment in segments[:midpoint])
    second_half_text = " ".join(segment.text for segment in segments[midpoint:])
    if "?" not in first_half_text:
        return 0.0
    answer_tokens = {"because", "so", "answer", "explanation", "reason"}
    has_connective = bool(set(_tokens(second_half_text)) & answer_tokens)
    has_payoff = score_payoff_signal(second_half_text) > 0
    if has_connective or has_payoff:
        return 1.0
    return 0.35


def score_motion_density(
    frames: list[VisionFrame], start_seconds: float, end_seconds: float
) -> float:
    scores = [
        frame.motion_score
        for frame in frames
        if start_seconds <= frame.timestamp_seconds < end_seconds
    ]
    if not scores:
        return 0.0
    return _clamp01(sum(scores) / len(scores))


def score_shot_change_density(
    frames: list[VisionFrame], start_seconds: float, end_seconds: float
) -> float:
    duration = max(end_seconds - start_seconds, 1e-6)
    changes = sum(
        1
        for frame in frames
        if start_seconds <= frame.timestamp_seconds < end_seconds and frame.shot_change
    )
    return _clamp01(changes / (duration / 5.0))


def score_face_presence(
    frames: list[VisionFrame], start_seconds: float, end_seconds: float
) -> float:
    in_window = [
        frame
        for frame in frames
        if start_seconds <= frame.timestamp_seconds < end_seconds
    ]
    if not in_window:
        return 0.0
    with_face = sum(1 for frame in in_window if frame.primary_face is not None)
    return _clamp01(with_face / len(in_window))


def score_speaker_interaction(segments: tuple[TranscriptSegment, ...]) -> float:
    if len(segments) < 2:
        return 0.0
    duration = max(segments[-1].end_seconds - segments[0].start_seconds, 1e-6)
    transitions = sum(
        1
        for previous, current in zip(segments, segments[1:], strict=False)
        if previous.speaker != current.speaker
    )
    return _clamp01(transitions / (duration / 10.0))


def score_delivery_variance(segments: tuple[TranscriptSegment, ...]) -> float:
    gaps = _inter_word_gaps(segments)
    if len(gaps) < 3:
        return 0.0
    try:
        stdev = statistics.pstdev(gaps)
    except statistics.StatisticsError:
        return 0.0
    return _clamp01(stdev / 0.35)


def has_buried_lead(first_segment_text: str) -> bool:
    lowered = first_segment_text.lower()
    return any(phrase in lowered for phrase in BURIED_LEAD_PHRASES)


def has_dangling_question(segments: tuple[TranscriptSegment, ...]) -> bool:
    if not segments:
        return False
    final_text = segments[-1].text.strip()
    if not final_text.endswith("?"):
        return False
    combined = " ".join(segment.text for segment in segments[:-1])
    return score_payoff_signal(combined) == 0.0


def is_rambling_middle(
    segments: tuple[TranscriptSegment, ...],
    frames: list[VisionFrame],
    *,
    motion_ceiling: float,
    keyword_floor: float,
) -> bool:
    if len(segments) < 3:
        return False
    middle = segments[1:-1]
    middle_text = " ".join(segment.text for segment in middle)
    motion = score_motion_density(
        frames, middle[0].start_seconds, middle[-1].end_seconds
    )
    keyword = score_keyword_spike(middle_text)
    return motion <= motion_ceiling and keyword <= keyword_floor


def derive_spike_categories(signals: WindowSignals) -> list[str]:
    categories: list[str] = []
    if signals.keyword >= 0.16:
        if signals.speaker_interaction >= 0.1:
            categories.append("emotional_confrontation")
        else:
            categories.append("controversy")
        if signals.payoff <= 0.1:
            categories.append("absurdity")
    if signals.motion >= 0.55 or signals.shot_change >= 0.5:
        categories.append("action")
    if signals.payoff >= 0.45 and signals.numeric >= 0.12:
        categories.append("unusually_useful_claim")
    seen: set[str] = set()
    deduped: list[str] = []
    for category in categories:
        if category in seen:
            continue
        seen.add(category)
        deduped.append(category)
    return deduped


def derive_reasons(signals: WindowSignals) -> list[str]:
    reasons: list[str] = []
    if signals.hook >= 0.45:
        reasons.append("strong hook")
    if signals.keyword >= 0.16:
        reasons.append("shareability")
    if signals.payoff >= 0.35:
        reasons.append("payoff")
    if signals.motion >= 0.55 or signals.shot_change >= 0.5:
        reasons.append("visual momentum")
    if signals.speaker_interaction >= 0.1:
        reasons.append("back-and-forth")
    if signals.numeric >= 0.2:
        reasons.append("numeric detail")
    if signals.question_to_answer >= 1.0:
        reasons.append("question answered")
    if not reasons:
        reasons.append("clarity")
    return reasons


def _score_window(
    segments: tuple[TranscriptSegment, ...],
    frames: list[VisionFrame],
    cfg: MiningConfig,
) -> ScoredWindow:
    start = segments[0].start_seconds
    end = segments[-1].end_seconds
    aggregate_text = " ".join(segment.text for segment in segments)
    first_text = segments[0].text

    signals = WindowSignals(
        hook=score_hook_strength(first_text),
        keyword=score_keyword_spike(aggregate_text),
        numeric=score_numeric_density(aggregate_text),
        interjection=score_interjection_density(aggregate_text),
        payoff=score_payoff_signal(aggregate_text),
        question_to_answer=score_question_to_answer(segments),
        motion=score_motion_density(frames, start, end),
        shot_change=score_shot_change_density(frames, start, end),
        face_presence=score_face_presence(frames, start, end),
        speaker_interaction=score_speaker_interaction(segments),
        delivery_variance=score_delivery_variance(segments),
        buried_lead=has_buried_lead(first_text),
        dangling_question=has_dangling_question(segments),
        rambling_middle=is_rambling_middle(
            segments,
            frames,
            motion_ceiling=cfg.rambling_motion_ceiling,
            keyword_floor=cfg.rambling_keyword_floor,
        ),
    )

    positive = (
        cfg.weights.base
        + cfg.weights.hook * signals.hook
        + cfg.weights.keyword * signals.keyword
        + cfg.weights.numeric * signals.numeric
        + cfg.weights.interjection * signals.interjection
        + cfg.weights.payoff * signals.payoff
        + cfg.weights.question_to_answer * signals.question_to_answer
        + cfg.weights.motion * signals.motion
        + cfg.weights.shot_change * signals.shot_change
        + cfg.weights.face_presence * signals.face_presence
        + cfg.weights.speaker_interaction * signals.speaker_interaction
        + cfg.weights.delivery_variance * signals.delivery_variance
    )
    penalty = 0.0
    if signals.buried_lead:
        penalty += cfg.penalties.buried_lead
    if signals.dangling_question:
        penalty += cfg.penalties.dangling_question
    if signals.rambling_middle:
        penalty += cfg.penalties.rambling_middle

    return ScoredWindow(
        segments=segments,
        signals=signals,
        score=_clamp01(positive - penalty),
    )


def _greedy_deduplicate(
    scored: list[ScoredWindow], *, max_overlap_ratio: float
) -> list[ScoredWindow]:
    kept: list[ScoredWindow] = []
    for candidate in scored:
        if any(
            _overlap_ratio(candidate, existing) > max_overlap_ratio for existing in kept
        ):
            continue
        kept.append(candidate)
    return kept


def _fill_minimum_candidates(
    selected: list[ScoredWindow],
    scored: list[ScoredWindow],
    *,
    minimum: int,
    max_overlap_ratio: float,
) -> list[ScoredWindow]:
    if len(selected) >= minimum:
        return selected
    filled = list(selected)
    selected_keys = {
        (window.start_seconds, window.end_seconds)
        for window in filled
    }
    for candidate in scored:
        key = (candidate.start_seconds, candidate.end_seconds)
        if key in selected_keys:
            continue
        if any(
            _overlap_ratio(candidate, existing) > max_overlap_ratio
            for existing in filled
        ):
            continue
        filled.append(candidate)
        selected_keys.add(key)
        if len(filled) >= minimum:
            break
    if len(filled) >= minimum:
        return filled
    for candidate in scored:
        key = (candidate.start_seconds, candidate.end_seconds)
        if key in selected_keys:
            continue
        filled.append(candidate)
        selected_keys.add(key)
        if len(filled) >= minimum:
            break
    return filled


def _overlap_ratio(a: ScoredWindow, b: ScoredWindow) -> float:
    overlap = max(
        0.0, min(a.end_seconds, b.end_seconds) - max(a.start_seconds, b.start_seconds)
    )
    shorter = min(a.end_seconds - a.start_seconds, b.end_seconds - b.start_seconds)
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def _inter_word_gaps(segments: tuple[TranscriptSegment, ...]) -> list[float]:
    words: list[TranscriptWord] = [
        word for segment in segments for word in segment.words
    ]
    gaps: list[float] = []
    for previous, current in zip(words, words[1:], strict=False):
        gap = current.start_seconds - previous.end_seconds
        if gap >= 0:
            gaps.append(gap)
    return gaps


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


def _clamp01(value: float) -> float:
    if value != value:
        return 0.0
    return min(max(value, 0.0), 1.0)
