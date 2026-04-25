from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from clippos.models.media import AspectRatio, ContractModel

SceneMode = Literal["TRACK", "GENERAL"]

CaptionPreset = Literal[
    "hook-default",
    "bottom-creator",
    "bottom-compact",
    "lower-third-clean",
    "center-punch",
    "top-clean",
]

DEFAULT_CAPTION_PRESET: CaptionPreset = "hook-default"


class CaptionWord(ContractModel):
    text: str
    start_seconds: Annotated[float, Field(ge=0)]
    end_seconds: Annotated[float, Field(ge=0)]
    emphasis: bool = False

    @model_validator(mode="after")
    def validate_time_bounds(self) -> "CaptionWord":
        if self.end_seconds < self.start_seconds:
            raise ValueError(
                "end_seconds must be greater than or equal to start_seconds"
            )
        return self


class CaptionLine(ContractModel):
    start_seconds: Annotated[float, Field(ge=0)]
    end_seconds: Annotated[float, Field(ge=0)]
    text: str
    words: list[CaptionWord]

    @model_validator(mode="after")
    def validate_time_bounds(self) -> "CaptionLine":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class CropAnchor(ContractModel):
    timestamp_seconds: Annotated[float, Field(ge=0)]
    center_x: Annotated[float, Field(ge=0, le=1)]
    center_y: Annotated[float, Field(ge=0, le=1)]


class CropPlan(ContractModel):
    aspect_ratio: AspectRatio
    source_width: Annotated[int, Field(gt=0)]
    source_height: Annotated[int, Field(gt=0)]
    target_width: Annotated[int, Field(gt=0)]
    target_height: Annotated[int, Field(gt=0)]
    anchors: list[CropAnchor]

    @model_validator(mode="after")
    def validate_crop_plan(self) -> "CropPlan":
        if not self.anchors:
            raise ValueError("anchors must not be empty")
        if self.target_width > self.source_width:
            raise ValueError("target_width must be less than or equal to source_width")
        if self.target_height > self.source_height:
            raise ValueError(
                "target_height must be less than or equal to source_height"
            )
        return self


class RenderManifest(ContractModel):
    clip_id: str
    approved: bool
    source_video: Path
    start_seconds: Annotated[float, Field(ge=0)]
    end_seconds: Annotated[float, Field(ge=0)]
    outputs: dict[AspectRatio, Path]
    crop_plans: dict[AspectRatio, CropPlan]
    caption_plan: list[CaptionLine]
    mode: SceneMode = "TRACK"
    caption_preset: CaptionPreset = DEFAULT_CAPTION_PRESET

    @model_validator(mode="after")
    def validate_bounds_and_keys(self) -> "RenderManifest":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        if set(self.outputs.keys()) != set(self.crop_plans.keys()):
            raise ValueError("outputs and crop_plans must cover the same ratios")
        for ratio, plan in self.crop_plans.items():
            if plan.aspect_ratio != ratio:
                raise ValueError(
                    f"crop_plans[{ratio!r}].aspect_ratio must equal {ratio!r}"
                )
        return self
