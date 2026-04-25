from pathlib import Path
from typing import Annotated

from pydantic import Field

from clipper.models.media import AspectRatio, ContractModel
from clipper.models.render import DEFAULT_CAPTION_PRESET, CaptionPreset


class OutputProfile(ContractModel):
    ratios: list[AspectRatio] = Field(default_factory=lambda: ["9:16", "1:1", "16:9"])
    caption_preset: CaptionPreset = DEFAULT_CAPTION_PRESET
    # v1.1 (docs/v1.1.md): when true (default), the orchestrator runs a
    # `brief` stage between mine and score that asks the harness model
    # to author a one-paragraph VideoBrief from the transcript. The
    # brief is attached to scoring + packaging requests as additional
    # context. Set false to skip the extra model handoff; per-clip
    # scoring still works, just without the global frame.
    video_brief: bool = True


class ClipperJob(ContractModel):
    video_path: Path
    output_dir: Path
    review_required: bool = True
    output_profile: OutputProfile = Field(default_factory=OutputProfile)
    max_candidates: Annotated[int, Field(gt=0)] = 12
