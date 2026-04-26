from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from clippos.models.media import ContractModel

SpikeCategory = Literal[
    "emotional_confrontation",
    "controversy",
    "taboo",
    "absurdity",
    "action",
    "unusually_useful_claim",
    # M4 (docs/miner-quality.md) — interview / podcast vertical.
    "expert_endorsement",
    "specific_pick",
    # Resurfaced from earlier work that was reset away during the
    # 2026-04-25 sync. Numeric / market-move hooks are clip-worthy
    # enough to deserve their own category.
    "big_number",
]

PenaltyCategory = Literal[
    "buried_lead",
    "dangling_question",
    "rambling_middle",
    "context_dependent",
    "low_delivery",
]

SPIKE_CATEGORIES: tuple[SpikeCategory, ...] = (
    "emotional_confrontation",
    "controversy",
    "taboo",
    "absurdity",
    "action",
    "unusually_useful_claim",
    "expert_endorsement",
    "specific_pick",
    "big_number",
)

PENALTY_CATEGORIES: tuple[PenaltyCategory, ...] = (
    "buried_lead",
    "dangling_question",
    "rambling_middle",
    "context_dependent",
    "low_delivery",
)

VideoBriefPattern = Annotated[str, Field(min_length=1)]


class MiningSignals(ContractModel):
    hook: Annotated[float, Field(ge=0, le=1)]
    keyword: Annotated[float, Field(ge=0, le=1)]
    numeric: Annotated[float, Field(ge=0, le=1)]
    interjection: Annotated[float, Field(ge=0, le=1)]
    payoff: Annotated[float, Field(ge=0, le=1)]
    question_to_answer: Annotated[float, Field(ge=0, le=1)]
    motion: Annotated[float, Field(ge=0, le=1)]
    shot_change: Annotated[float, Field(ge=0, le=1)]
    face_presence: Annotated[float, Field(ge=0, le=1)]
    speaker_interaction: Annotated[float, Field(ge=0, le=1)]
    delivery_variance: Annotated[float, Field(ge=0, le=1)]
    buried_lead: bool
    dangling_question: bool
    rambling_middle: bool
    reasons: list[str]
    spike_categories: list[str]


class ClipBrief(ContractModel):
    clip_id: str
    clip_hash: str
    start_seconds: Annotated[float, Field(ge=0)]
    end_seconds: Annotated[float, Field(ge=0)]
    duration_seconds: Annotated[float, Field(gt=0)]
    transcript: str
    speakers: list[str]
    mining_score: Annotated[float, Field(ge=0, le=1)]
    mining_signals: MiningSignals

    @model_validator(mode="after")
    def validate_time_bounds(self) -> "ClipBrief":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class VideoBrief(ContractModel):
    """The harness model's pre-scoring synthesis of what makes THIS video
    clippable (v1.1, see docs/v1.1.md).

    Authored once per video, before per-clip scoring. Attached to the
    scoring + packaging requests as additional context so per-clip
    judgement can be aware of the global thesis. Cached per workspace
    so re-running scoring does not re-pay the brief cost.
    """

    rubric_version: str
    job_id: str
    theme: Annotated[str, Field(min_length=1)]
    video_format: Annotated[str, Field(min_length=1)]
    expected_viral_patterns: Annotated[
        list[VideoBriefPattern], Field(min_length=3, max_length=5)
    ]
    anti_patterns: Annotated[
        list[VideoBriefPattern], Field(max_length=3)
    ] = Field(default_factory=list)
    audience: str | None = None
    tone: str | None = None
    notes: str | None = None


class VideoBriefRequest(ContractModel):
    """Handoff payload sent to the harness model to author a VideoBrief."""

    rubric_version: str
    job_id: str
    video_path: Path
    transcript_excerpt: str
    transcript_truncated: bool
    duration_seconds: Annotated[float, Field(ge=0)]
    speakers: list[str]
    brief_prompt: str
    response_schema: dict


class VideoBriefResponse(ContractModel):
    """Wrapper around the brief produced by the harness model. Mirrors
    ScoringResponse / PackageResponse shape so callers can use the same
    rubric_version / job_id integrity checks."""

    rubric_version: str
    job_id: str
    brief: VideoBrief


class ScoringRequest(ContractModel):
    rubric_version: str
    job_id: str
    video_path: Path
    rubric_prompt: str
    response_schema: dict
    clips: list[ClipBrief]
    # v1.1: optional global frame produced by the brief stage. None when
    # the brief stage was skipped (output_profile.video_brief=False) or
    # when the harness has not yet produced the brief response. Old
    # scoring-request.json files predating v1.1 lack this field; the
    # default keeps them loadable.
    video_brief: VideoBrief | None = None


class RubricScores(ContractModel):
    hook: Annotated[float, Field(ge=0, le=1)]
    shareability: Annotated[float, Field(ge=0, le=1)]
    standalone_clarity: Annotated[float, Field(ge=0, le=1)]
    payoff: Annotated[float, Field(ge=0, le=1)]
    delivery_energy: Annotated[float, Field(ge=0, le=1)]
    quotability: Annotated[float, Field(ge=0, le=1)]


class ClipScore(ContractModel):
    clip_id: str
    clip_hash: str
    rubric: RubricScores
    spike_categories: list[SpikeCategory]
    penalties: list[PenaltyCategory]
    final_score: Annotated[float, Field(ge=0, le=1)]
    title: str
    hook: str
    reasons: list[str]


class ScoringResponse(ContractModel):
    rubric_version: str
    job_id: str
    scores: list[ClipScore]

    @model_validator(mode="after")
    def validate_no_duplicate_clip_identifiers(self) -> "ScoringResponse":
        seen_ids: set[str] = set()
        seen_hashes: set[str] = set()
        duplicate_ids: list[str] = []
        duplicate_hashes: list[str] = []
        for score in self.scores:
            if score.clip_id in seen_ids and score.clip_id not in duplicate_ids:
                duplicate_ids.append(score.clip_id)
            else:
                seen_ids.add(score.clip_id)
            if (
                score.clip_hash in seen_hashes
                and score.clip_hash not in duplicate_hashes
            ):
                duplicate_hashes.append(score.clip_hash)
            else:
                seen_hashes.add(score.clip_hash)
        if duplicate_ids:
            raise ValueError(
                "scores contains duplicate clip_id entries: "
                + ", ".join(repr(v) for v in duplicate_ids)
            )
        if duplicate_hashes:
            raise ValueError(
                "scores contains duplicate clip_hash entries: "
                + ", ".join(repr(v) for v in duplicate_hashes)
            )
        return self
