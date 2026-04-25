from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from clipper.models.media import ContractModel

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


class ScoringRequest(ContractModel):
    rubric_version: str
    job_id: str
    video_path: Path
    rubric_prompt: str
    response_schema: dict
    clips: list[ClipBrief]


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
