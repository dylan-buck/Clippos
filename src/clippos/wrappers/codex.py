from pathlib import Path

from clippos.models.job import ClipposJob
from clippos.models.scoring import ScoringRequest, ScoringResponse
from clippos.wrappers.common import (
    build_common_job,
    load_workspace_scoring_request,
    write_workspace_scoring_response,
)


def codex_job_from_args(video_path: str, output_dir: str) -> ClipposJob:
    return build_common_job(video_path, output_dir)


def codex_load_scoring_request(workspace_dir: Path) -> ScoringRequest:
    return load_workspace_scoring_request(workspace_dir)


def codex_write_scoring_response(
    workspace_dir: Path, response: ScoringResponse | dict
) -> Path:
    return write_workspace_scoring_response(workspace_dir, response)
