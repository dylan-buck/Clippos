from __future__ import annotations

import re
import statistics
import sys
from dataclasses import dataclass, field

from clippos.models.candidate import CandidateClip
from clippos.pipeline.transcribe import (
    TranscriptSegment,
    TranscriptTimeline,
    TranscriptWord,
)
from clippos.pipeline.vision import VisionFrame, VisionTimeline

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

# M3 (docs/miner-quality.md): interview-specific phrases tuned to surface
# expert-Q&A and stock-pick / endorsement moments that monologue
# keywords miss entirely. Activated only when speaker_interaction is
# high (gated in `_score_window`) so they do not pollute solo-monologue
# scoring. The dogfood video had "Bloom Energy... hands down one of the
# best investors of all time" go undetected because the host-only
# keyword buckets do not match interview vocabulary.
INTERVIEW_KEYWORD_PHRASES = (
    "the play here",
    "what's your take",
    "what is your take",
    "the most important thing",
    "if you want to follow",
    "hands down",
    "one of the best",
    "the way i think about",
    "the way i look at",
    "my biggest position",
    "biggest position",
    "the trade i like",
    "the trade i want",
    "i'm long",
    "i am long",
    "i'm short",
    "i am short",
    "i'm a holder",
    "im a holder",
    "huge holder",
    "massive holder",
    "best investor",
    "smartest guy",
    "smartest people",
    "the secret sauce",
    "playbook",
    "the call here",
    "high conviction",
)
INTERVIEW_KEYWORDS = frozenset(
    {
        "long",
        "short",
        "position",
        "holder",
        "ticker",
        "playbook",
        "conviction",
        "endorse",
        "endorsed",
        "endorsement",
        "alpha",
        "moat",
        "thesis",
        "compounding",
        "compounder",
        "fundamentals",
        "valuation",
        "multiple",
    }
)
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


def _status(message: str) -> None:
    print(f"[clippos] {message}", file=sys.stderr, flush=True)


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
    # M1 (docs/miner-quality.md): bumped 0.05 -> 0.12 to give multi-speaker
    # exchanges parity with hook/keyword/payoff. The 2026-04-25 dogfood
    # surfaced that an entire 8-min guest interview block was invisible to
    # mining because the dominant signals (hook + keyword + payoff = 0.36
    # combined weight) are calibrated for solo monologue patterns; a single
    # 0.05 weight on multi-speaker activity could not compete. Interview /
    # podcast / guest-Q&A content is a primary use case, so multi-speaker
    # signal deserves the same weight as the other top-tier signals.
    speaker_interaction: float = 0.12
    delivery_variance: float = 0.03
    # M3 (docs/miner-quality.md): keyword-bucket signal gated on
    # multi-speaker activity, so monologue scoring is unchanged. Same
    # weight as the monologue keyword bucket — interview vocabulary
    # ("hands down", "the play here", "long $TICKER") is just as
    # clip-worthy as monologue vocabulary ("crazy", "insane") in its
    # respective context.
    interview_keyword: float = 0.12


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
    # M5 (docs/miner-quality.md): a lower score floor applied inside
    # detected interview blocks. A 0.30 score on a guest stock pick is
    # clip-worthy in interview context; the same score on solo monologue
    # rambling probably is not. Used by the M2 representation guarantee
    # below to keep multi-speaker windows that didn't clear the regular
    # score_floor but are the best (or only) candidate from their block.
    multi_speaker_score_floor: float = 0.20
    # M2 thresholds — what counts as an "interview block" worth
    # guaranteeing representation for.
    interview_block_min_seconds: float = 30.0
    interview_block_min_transitions: int = 2


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
    # M3: gated keyword signal — only non-zero when multi-speaker
    # activity is present. See INTERVIEW_KEYWORD_PHRASES /
    # INTERVIEW_KEYWORDS for the interview-tuned vocabulary.
    interview_keyword: float
    buried_lead: bool
    dangling_question: bool
    rambling_middle: bool


@dataclass(frozen=True)
class ScoredWindow:
    segments: tuple[TranscriptSegment, ...]
    signals: WindowSignals
    score: float
    visual_summary: dict[str, object] | None = None

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
    _status(f"Mining: scoring {len(windows)} transcript windows.")
    scored: list[ScoredWindow] = [
        _score_window(window, vision_timeline.frames, cfg) for window in windows
    ]
    scored.sort(key=lambda sw: sw.score, reverse=True)
    above_floor = [sw for sw in scored if sw.score >= cfg.score_floor]
    selected = _greedy_deduplicate(
        above_floor, max_overlap_ratio=cfg.max_overlap_ratio
    )

    # M2 + M5 (docs/miner-quality.md): guarantee at least one
    # representative window per detected interview block. Without this,
    # the dogfood video had an entire 8-min guest interview block produce
    # zero candidates — every window from that block scored just below
    # the regular 0.35 floor and got discarded before backfill ran on
    # the wider corpus. Detection runs on the actual transcript
    # (independent of windowing) so blocks are found even when no single
    # window inside them scored well.
    interview_blocks = _detect_interview_blocks(
        transcript_timeline.segments,
        min_duration_seconds=cfg.interview_block_min_seconds,
        min_transitions=cfg.interview_block_min_transitions,
    )
    if interview_blocks:
        selected = _ensure_interview_block_representation(
            selected,
            scored,
            interview_blocks,
            multi_speaker_floor=cfg.multi_speaker_score_floor,
            max_overlap_ratio=cfg.max_overlap_ratio,
        )

    minimum = min(max_candidates, max(cfg.min_candidates, 0))
    if len(selected) < minimum:
        selected = _fill_minimum_candidates(
            selected,
            scored,
            minimum=minimum,
            max_overlap_ratio=cfg.max_overlap_ratio,
        )
    _status(
        "Mining: selected "
        f"{min(len(selected), max_candidates)} candidate(s) "
        f"({len(above_floor)} above score floor, target max {max_candidates})."
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


def score_interview_keyword_spike(text: str) -> float:
    """M3 (docs/miner-quality.md): score interview-tuned vocabulary.

    Returns 0.0 when there are no hits; otherwise scores in [0, 1] based
    on combined phrase + token presence. The caller in `_score_window`
    gates the contribution on `speaker_interaction >= 0.1` so this signal
    only fires for multi-speaker windows — it does not apply to solo
    monologue content where these phrases would mean something different
    or be coincidental.
    """
    lowered = text.lower()
    phrase_hits = sum(1 for phrase in INTERVIEW_KEYWORD_PHRASES if phrase in lowered)
    tokens = set(_tokens(lowered))
    keyword_hits = len(tokens & INTERVIEW_KEYWORDS)
    if phrase_hits == 0 and keyword_hits == 0:
        return 0.0
    # Phrases are stronger signals than single tokens (a token like "long"
    # appears in non-trading contexts; the phrase "i'm long" is unambiguous).
    return _clamp01(0.45 * min(phrase_hits, 2) + 0.2 * min(keyword_hits, 3))


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
    # M4 (docs/miner-quality.md): interview-vertical spike categories.
    # `interview_keyword` is already gated on speaker_interaction, so we
    # don't need to re-check it here — but we do require at least a
    # non-trivial signal to suppress noise from one-off mentions.
    if signals.interview_keyword >= 0.2:
        categories.append("expert_endorsement")
    if signals.interview_keyword >= 0.1 and signals.numeric >= 0.1:
        categories.append("specific_pick")
    # `big_number` — concrete numeric hooks, regardless of speaker mix.
    # High numeric density alone is a strong proxy for the "$100B in a
    # day" / "lost $40k" pattern. The model still validates that the
    # number is the centerpiece (this is just a hint).
    if signals.numeric >= 0.4:
        categories.append("big_number")
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

    speaker_interaction = score_speaker_interaction(segments)
    # M3: interview-keyword signal only contributes when the window has
    # actual multi-speaker activity. Avoids polluting solo monologue
    # scoring with words like "long" / "position" that mean something
    # different outside of an interview-trader context.
    interview_keyword = (
        score_interview_keyword_spike(aggregate_text)
        if speaker_interaction >= 0.1
        else 0.0
    )

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
        speaker_interaction=speaker_interaction,
        delivery_variance=score_delivery_variance(segments),
        interview_keyword=interview_keyword,
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
        + cfg.weights.interview_keyword * signals.interview_keyword
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
        visual_summary=_summarize_visual_context(frames, start, end),
    )


def _summarize_visual_context(
    frames: list[VisionFrame], start_seconds: float, end_seconds: float
) -> dict[str, object]:
    in_window = [
        frame
        for frame in frames
        if start_seconds <= frame.timestamp_seconds < end_seconds
    ]
    duration = max(end_seconds - start_seconds, 1e-6)
    if not in_window:
        return {
            "frame_count": 0,
            "face_presence_ratio": 0.0,
            "avg_motion": 0.0,
            "peak_motion": 0.0,
            "shot_change_count": 0,
            "shot_change_rate_per_minute": 0.0,
            "avg_face_center_x": None,
            "avg_face_center_y": None,
            "description": "No sampled vision frames landed inside this clip window.",
        }

    motions = [frame.motion_score for frame in in_window]
    face_frames = [frame for frame in in_window if frame.primary_face is not None]
    shot_change_count = sum(1 for frame in in_window if frame.shot_change)
    face_ratio = len(face_frames) / len(in_window)
    avg_motion = sum(motions) / len(motions)
    peak_motion = max(motions)
    avg_face_center_x = (
        sum(frame.primary_face.center_x for frame in face_frames) / len(face_frames)
        if face_frames
        else None
    )
    avg_face_center_y = (
        sum(frame.primary_face.center_y for frame in face_frames) / len(face_frames)
        if face_frames
        else None
    )
    description = (
        f"faces in {face_ratio:.0%} of sampled frames; "
        f"avg motion {avg_motion:.2f}, peak motion {peak_motion:.2f}; "
        f"{shot_change_count} shot change(s)"
    )
    return {
        "frame_count": len(in_window),
        "face_presence_ratio": round(_clamp01(face_ratio), 4),
        "avg_motion": round(_clamp01(avg_motion), 4),
        "peak_motion": round(_clamp01(peak_motion), 4),
        "shot_change_count": shot_change_count,
        "shot_change_rate_per_minute": round(shot_change_count / (duration / 60.0), 4),
        "avg_face_center_x": (
            round(_clamp01(avg_face_center_x), 4)
            if avg_face_center_x is not None
            else None
        ),
        "avg_face_center_y": (
            round(_clamp01(avg_face_center_y), 4)
            if avg_face_center_y is not None
            else None
        ),
        "description": description,
    }


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


@dataclass(frozen=True)
class InterviewBlock:
    """A contiguous transcript stretch with substantial multi-speaker
    activity (M2 in docs/miner-quality.md).

    The block boundaries are inclusive segment-time bounds. The block is
    used purely for selection purposes — it does not change scoring or
    spike-category derivation, only ensures at least one window from
    inside it survives candidate selection."""
    start_seconds: float
    end_seconds: float
    speakers: frozenset[str]
    transition_count: int


def _detect_interview_blocks(
    segments: list[TranscriptSegment],
    *,
    min_duration_seconds: float,
    min_transitions: int,
) -> list[InterviewBlock]:
    """Find contiguous stretches with substantial multi-speaker activity.

    A "block" is a contiguous run of segments where:
      - duration >= min_duration_seconds, and
      - the run contains at least min_transitions speaker changes, and
      - at least two distinct speakers participate.

    Blocks merge adjacent multi-speaker activity even when separated by
    short single-speaker gaps (<10s) so the long-run guest interview
    block in the dogfood video is one block, not many fragments.
    """
    if len(segments) < 2:
        return []
    # First pass: find every speaker transition.
    transition_indices = [
        i
        for i in range(1, len(segments))
        if segments[i].speaker != segments[i - 1].speaker
    ]
    if not transition_indices:
        return []

    # Greedy block builder: walk transitions, accumulate them into a
    # current run as long as adjacent transitions are within `merge_gap`
    # seconds of each other. This collapses Q-A-Q-A patterns into one
    # block instead of fragmenting on every back-and-forth.
    merge_gap = 10.0
    runs: list[list[int]] = []
    current: list[int] = []
    for idx in transition_indices:
        if not current:
            current = [idx]
            continue
        prev_idx = current[-1]
        gap = segments[idx].start_seconds - segments[prev_idx].end_seconds
        if gap <= merge_gap:
            current.append(idx)
        else:
            runs.append(current)
            current = [idx]
    if current:
        runs.append(current)

    blocks: list[InterviewBlock] = []
    # Only include a bookend segment (the speaker immediately before the
    # first transition / after the last) when it sits within this many
    # seconds of the alternation. Otherwise we accidentally pull in a
    # whole adjacent monologue and the block range bleeds into solo
    # content. The dogfood test data had a 10s gap between the host
    # monologue and the guest's arrival — the bookend logic was
    # absorbing the whole prior monologue into the "interview block",
    # which then satisfied the M2 guarantee with a host-only window.
    bookend_max_gap_seconds = 5.0
    for run in runs:
        if len(run) < min_transitions:
            continue
        start_idx = run[0]
        if start_idx > 0:
            gap = segments[start_idx].start_seconds - segments[start_idx - 1].end_seconds
            if gap <= bookend_max_gap_seconds:
                start_idx -= 1
        end_idx = run[-1]
        if end_idx < len(segments) - 1:
            gap = segments[end_idx + 1].start_seconds - segments[end_idx].end_seconds
            if gap <= bookend_max_gap_seconds:
                end_idx += 1
        block_segments = segments[start_idx : end_idx + 1]
        duration = (
            block_segments[-1].end_seconds - block_segments[0].start_seconds
        )
        if duration < min_duration_seconds:
            continue
        speakers = frozenset(
            seg.speaker for seg in block_segments if seg.speaker
        )
        if len(speakers) < 2:
            continue
        blocks.append(
            InterviewBlock(
                start_seconds=block_segments[0].start_seconds,
                end_seconds=block_segments[-1].end_seconds,
                speakers=speakers,
                transition_count=len(run),
            )
        )
    return blocks


def _ensure_interview_block_representation(
    selected: list[ScoredWindow],
    scored: list[ScoredWindow],
    blocks: list[InterviewBlock],
    *,
    multi_speaker_floor: float,
    max_overlap_ratio: float,
) -> list[ScoredWindow]:
    """For each detected interview block, guarantee at least one
    representative scored window enters the final selection.

    The 2026-04-25 dogfood produced zero candidates from an 8-minute
    guest interview because every window inside it scored just below
    the regular `score_floor` (0.35). This pass relaxes the floor to
    `multi_speaker_floor` (default 0.20) for windows that overlap a
    detected block, picks the best one not already represented, and
    inserts it into `selected` while still respecting the dedup ratio
    against existing selections.
    """
    if not blocks:
        return selected
    enriched = list(selected)
    for block in blocks:
        if _block_already_represented(block, enriched):
            continue
        # Find the highest-scoring window that overlaps this block AND
        # clears the relaxed floor. `scored` is already sorted descending.
        appended = False
        for candidate in scored:
            if candidate.score < multi_speaker_floor:
                # Sorted desc — nothing remaining can clear the floor.
                break
            if not _window_overlaps_block(candidate, block):
                continue
            if any(
                _overlap_ratio(candidate, existing) > max_overlap_ratio
                for existing in enriched
            ):
                continue
            enriched.append(candidate)
            appended = True
            break
        if appended:
            continue
        # If still no representative, fall back to any overlapping window
        # that doesn't dedup-collide. Better to surface a marginal clip
        # the model can reject than to leave the block invisible. (Note:
        # the `for...else` idiom doesn't work here because both the
        # below-floor break and the success break would skip the else.)
        for candidate in scored:
            if not _window_overlaps_block(candidate, block):
                continue
            if any(
                _overlap_ratio(candidate, existing) > max_overlap_ratio
                for existing in enriched
            ):
                continue
            enriched.append(candidate)
            break
    return enriched


def _block_already_represented(
    block: InterviewBlock, selected: list[ScoredWindow]
) -> bool:
    return any(_window_overlaps_block(window, block) for window in selected)


def _window_overlaps_block(window: ScoredWindow, block: InterviewBlock) -> bool:
    overlap = max(
        0.0,
        min(window.end_seconds, block.end_seconds)
        - max(window.start_seconds, block.start_seconds),
    )
    return overlap > 0.0


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
