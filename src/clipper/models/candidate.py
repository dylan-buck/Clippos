from typing import Annotated

from pydantic import Field, model_validator

from clipper.models.media import ContractModel


class CandidateClip(ContractModel):
    clip_id: str
    start_seconds: Annotated[float, Field(ge=0)]
    end_seconds: Annotated[float, Field(ge=0)]
    score: Annotated[float, Field(ge=0, le=1)]
    reasons: list[str]
    spike_categories: list[str]
    title: str = ""
    hook: str = ""

    @model_validator(mode="after")
    def validate_time_bounds(self) -> "CandidateClip":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self
