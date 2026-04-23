from pathlib import Path
from typing import Annotated

from pydantic import Field

from clipper.models.media import AspectRatio, ContractModel


class OutputProfile(ContractModel):
    ratios: list[AspectRatio] = Field(
        default_factory=lambda: ["9:16", "1:1", "16:9"]
    )
    caption_preset: str = "hook-default"


class ClipperJob(ContractModel):
    video_path: Path
    output_dir: Path
    review_required: bool = True
    output_profile: OutputProfile = Field(default_factory=OutputProfile)
    max_candidates: Annotated[int, Field(gt=0)] = 12
