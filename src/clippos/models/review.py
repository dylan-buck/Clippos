from pathlib import Path

from clippos.models.candidate import CandidateClip
from clippos.models.media import ContractModel


class ReviewManifest(ContractModel):
    job_id: str
    video_path: Path
    candidates: list[CandidateClip]
