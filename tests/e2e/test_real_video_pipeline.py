import json
import os
import shutil
from pathlib import Path

import pytest

from clipper.adapters.rubric import RUBRIC_VERSION
from clipper.adapters.storage import write_json
from clipper.models.job import ClipperJob
from clipper.models.scoring import ClipScore, RubricScores, ScoringResponse
from clipper.pipeline.orchestrator import run_job
from clipper.pipeline.scoring import (
    load_scoring_request,
    scoring_response_path,
)


@pytest.mark.e2e
def test_real_video_runs_mine_review_approval_and_render(tmp_path: Path) -> None:
    video_path = _require_real_video_path()
    _require_binary("ffmpeg")
    _require_binary("ffprobe")
    _require_engine_dependencies()
    _require_hf_token()

    job = ClipperJob(
        video_path=video_path,
        output_dir=tmp_path / "out",
        max_candidates=3,
    )

    request_path = run_job(job, stage="mine")
    workspace_dir = request_path.parent
    request = load_scoring_request(workspace_dir)
    assert request is not None
    assert request.clips, "real video produced no clip candidates"

    top_clip = request.clips[0]
    response = ScoringResponse(
        rubric_version=RUBRIC_VERSION,
        job_id=request.job_id,
        scores=[
            _score_for_clip(
                clip_id=clip.clip_id,
                clip_hash=clip.clip_hash,
                final_score=0.90 if clip.clip_id == top_clip.clip_id else 0.70,
            )
            for clip in request.clips
        ],
    )
    write_json(scoring_response_path(workspace_dir), response.model_dump(mode="json"))

    review_path = run_job(job, stage="review")
    review_payload = json.loads(review_path.read_text(encoding="utf-8"))
    for candidate in review_payload["candidates"]:
        candidate["approved"] = candidate["clip_id"] == top_clip.clip_id
    review_path.write_text(json.dumps(review_payload), encoding="utf-8")

    report_path = run_job(job, stage="render")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["job_id"] == request.job_id
    assert len(report["clips"]) == 1
    outputs = report["clips"][0]["outputs"]
    assert set(outputs) == {"9:16", "1:1", "16:9"}
    for ratio, relative_path in outputs.items():
        rendered = workspace_dir / relative_path
        assert rendered.exists(), f"{ratio} output was not written"
        assert rendered.stat().st_size > 0, f"{ratio} output is empty"


def _require_real_video_path() -> Path:
    raw_path = os.environ.get("CLIPPER_E2E_VIDEO")
    if not raw_path:
        pytest.skip("Set CLIPPER_E2E_VIDEO=/absolute/path/video.mp4 to run")
    video_path = Path(raw_path).expanduser().resolve()
    if not video_path.exists():
        pytest.fail(f"CLIPPER_E2E_VIDEO does not exist: {video_path}")
    return video_path


def _require_binary(binary: str) -> None:
    if shutil.which(binary) is None:
        pytest.skip(f"{binary} is required for real-video E2E")


def _require_engine_dependencies() -> None:
    pytest.importorskip("whisperx")
    pytest.importorskip("cv2")
    pytest.importorskip("retinaface")
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("scenedetect")


def _require_hf_token() -> None:
    if not any(
        os.environ.get(name)
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")
    ):
        pytest.skip("HF_TOKEN is required for WhisperX diarization")


def _score_for_clip(*, clip_id: str, clip_hash: str, final_score: float) -> ClipScore:
    return ClipScore(
        clip_id=clip_id,
        clip_hash=clip_hash,
        rubric=RubricScores(
            hook=0.85,
            shareability=0.80,
            standalone_clarity=0.80,
            payoff=0.75,
            delivery_energy=0.75,
            quotability=0.70,
        ),
        spike_categories=["controversy"],
        penalties=[],
        final_score=final_score,
        title="E2E selected clip",
        hook="Real video pipeline check",
        reasons=["selected by gated E2E scoring fixture"],
    )
