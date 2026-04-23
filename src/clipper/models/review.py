from pathlib import Path

from clipper.models.candidate import CandidateClip
from clipper.models.media import ContractModel


class ReviewManifest(ContractModel):
    job_id: str
    video_path: Path
    candidates: list[CandidateClip]
