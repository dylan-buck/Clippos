from pathlib import Path

from clipper.models.media import AspectRatio, ContractModel


class RenderManifest(ContractModel):
    clip_id: str
    approved: bool
    outputs: dict[AspectRatio, Path]
