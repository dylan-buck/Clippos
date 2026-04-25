from pathlib import Path

from clippos.adapters.storage import write_json
from clippos.models.job import ClipposJob
from clippos.models.scoring import ScoringRequest, ScoringResponse
from clippos.pipeline.scoring import (
    load_scoring_request,
    scoring_response_path,
)


def build_common_job(video_path: str, output_dir: str) -> ClipposJob:
    return ClipposJob.model_validate(
        {
            "video_path": video_path,
            "output_dir": output_dir,
        }
    )


def load_workspace_scoring_request(workspace_dir: Path) -> ScoringRequest:
    request = load_scoring_request(workspace_dir)
    if request is None:
        raise FileNotFoundError(
            f"No valid scoring-request.json found in {workspace_dir}"
        )
    return request


def write_workspace_scoring_response(
    workspace_dir: Path, response: ScoringResponse | dict
) -> Path:
    validated = (
        response
        if isinstance(response, ScoringResponse)
        else ScoringResponse.model_validate(response)
    )
    path = scoring_response_path(workspace_dir)
    write_json(path, validated.model_dump(mode="json"))
    return path
