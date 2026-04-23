from typing import Annotated

from pydantic import Field, model_validator

from clipper.models.media import ContractModel


class TranscriptWord(ContractModel):
    text: str
    start_seconds: Annotated[float, Field(ge=0)]
    end_seconds: Annotated[float, Field(ge=0)]
    confidence: Annotated[float, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def validate_time_bounds(self) -> "TranscriptWord":
        if self.end_seconds < self.start_seconds:
            raise ValueError("end_seconds must be greater than or equal to start_seconds")
        return self


class TranscriptSegment(ContractModel):
    speaker: str
    start_seconds: Annotated[float, Field(ge=0)]
    end_seconds: Annotated[float, Field(ge=0)]
    text: str
    words: list[TranscriptWord]

    @model_validator(mode="after")
    def validate_time_bounds(self) -> "TranscriptSegment":
        if self.end_seconds < self.start_seconds:
            raise ValueError("end_seconds must be greater than or equal to start_seconds")
        return self


class TranscriptTimeline(ContractModel):
    segments: list[TranscriptSegment]


class TranscriptPayload(ContractModel):
    segments: list[TranscriptSegment]


def build_transcript_timeline(payload: dict) -> TranscriptTimeline:
    validated_payload = TranscriptPayload.model_validate(payload)
    return TranscriptTimeline(segments=validated_payload.segments)
