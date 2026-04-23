from typing import Annotated

from pydantic import Field, Strict, StrictBool

from clipper.models.media import ContractModel


class FaceBox(ContractModel):
    center_x: Annotated[float, Strict(), Field(ge=0, le=1)]
    center_y: Annotated[float, Strict(), Field(ge=0, le=1)]
    width: Annotated[float, Strict(), Field(ge=0, le=1)]
    height: Annotated[float, Strict(), Field(ge=0, le=1)]
    confidence: Annotated[float, Strict(), Field(ge=0, le=1)]


class VisionFrame(ContractModel):
    timestamp_seconds: Annotated[float, Strict(), Field(ge=0)]
    motion_score: Annotated[float, Strict(), Field(ge=0, le=1)]
    shot_change: StrictBool
    primary_face: FaceBox | None = None


class VisionTimeline(ContractModel):
    frames: list[VisionFrame]


class VisionPayload(ContractModel):
    frames: list[VisionFrame]


def build_vision_timeline(payload: dict) -> VisionTimeline:
    validated_payload = VisionPayload.model_validate(payload)
    return VisionTimeline(frames=validated_payload.frames)
