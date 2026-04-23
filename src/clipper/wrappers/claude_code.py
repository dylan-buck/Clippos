from pathlib import Path

from clipper.models.job import ClipperJob
from clipper.models.scoring import ScoringRequest, ScoringResponse
from clipper.wrappers.common import (
    build_common_job,
    load_workspace_scoring_request,
    write_workspace_scoring_response,
)


def claude_job_from_args(video_path: str, output_dir: str) -> ClipperJob:
    return build_common_job(video_path, output_dir)


def claude_load_scoring_request(workspace_dir: Path) -> ScoringRequest:
    return load_workspace_scoring_request(workspace_dir)


def claude_write_scoring_response(
    workspace_dir: Path, response: ScoringResponse | dict
) -> Path:
    return write_workspace_scoring_response(workspace_dir, response)
