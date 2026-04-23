import json
import shutil
import subprocess
from pathlib import Path

import pytest

from clipper.adapters import ffmpeg_render
from clipper.adapters.rubric import RUBRIC_VERSION
from clipper.adapters.storage import write_json
from clipper.models.job import ClipperJob
from clipper.models.scoring import ClipScore, RubricScores, ScoringResponse
from clipper.pipeline.orchestrator import run_job
from clipper.pipeline.scoring import (
    load_scoring_request,
    scoring_response_path,
)


def test_smoke_mine_review_approve_render_with_real_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg smoke test requires ffmpeg and ffprobe on PATH")

    source_video = tmp_path / "source.mp4"
    output_dir = tmp_path / "out"
    _write_synthetic_video(source_video)

    monkeypatch.setattr(
        "clipper.pipeline.orchestrator.transcribe_video",
        lambda _path, _workspace: _transcript_payload(),
    )
    monkeypatch.setattr(
        "clipper.pipeline.orchestrator.analyze_video",
        lambda _path, _workspace: _vision_payload(),
    )
    monkeypatch.setitem(ffmpeg_render.CANONICAL_OUTPUT_DIMS, "9:16", (180, 320))
    monkeypatch.setitem(ffmpeg_render.CANONICAL_OUTPUT_DIMS, "1:1", (240, 240))
    monkeypatch.setitem(ffmpeg_render.CANONICAL_OUTPUT_DIMS, "16:9", (320, 180))

    job = ClipperJob(
        video_path=source_video,
        output_dir=output_dir,
        max_candidates=1,
    )

    request_path = run_job(job, stage="mine")
    workspace_dir = request_path.parent
    request = load_scoring_request(workspace_dir)
    assert request is not None
    assert request.clips

    brief = request.clips[0]
    response = ScoringResponse(
        rubric_version=RUBRIC_VERSION,
        job_id=request.job_id,
        scores=[
            ClipScore(
                clip_id=brief.clip_id,
                clip_hash=brief.clip_hash,
                rubric=RubricScores(
                    hook=0.9,
                    shareability=0.8,
                    standalone_clarity=0.8,
                    payoff=0.8,
                    delivery_energy=0.7,
                    quotability=0.7,
                ),
                spike_categories=["controversy", "absurdity"],
                penalties=[],
                final_score=0.86,
                title="The wild secret payoff",
                hook="Nobody talks about this part",
                reasons=["clear hook", "visual motion", "strong payoff"],
            )
        ],
    )
    write_json(scoring_response_path(workspace_dir), response.model_dump(mode="json"))

    review_path = run_job(job, stage="review")
    review_payload = json.loads(review_path.read_text(encoding="utf-8"))
    review_payload["candidates"][0]["approved"] = True
    review_path.write_text(json.dumps(review_payload), encoding="utf-8")

    report_path = run_job(job, stage="render")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_path.name == "render-report.json"
    assert len(report["clips"]) == 1
    outputs = report["clips"][0]["outputs"]
    assert set(outputs.keys()) == {"9:16", "1:1", "16:9"}
    for relative_path in outputs.values():
        rendered = workspace_dir / relative_path
        assert rendered.exists()
        assert rendered.stat().st_size > 0


def _write_synthetic_video(path: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x90:rate=5",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:sample_rate=48000",
        "-t",
        "16",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    subprocess.run(command, check=True)


def _transcript_payload() -> dict:
    return {
        "segments": [
            {
                "speaker": "speaker_1",
                "start_seconds": 0.0,
                "end_seconds": 8.0,
                "text": "Here's the thing, nobody talks about this secret scam.",
                "words": [],
            },
            {
                "speaker": "speaker_1",
                "start_seconds": 8.0,
                "end_seconds": 16.0,
                "text": "Turns out the result was wild and the payoff was instant.",
                "words": [],
            },
        ]
    }


def _vision_payload() -> dict:
    return {
        "frames": [
            {
                "timestamp_seconds": timestamp,
                "motion_score": 0.75,
                "shot_change": timestamp == 8.0,
                "primary_face": {
                    "center_x": 0.52,
                    "center_y": 0.45,
                    "width": 0.25,
                    "height": 0.45,
                    "confidence": 0.9,
                },
            }
            for timestamp in (0.0, 4.0, 8.0, 12.0, 16.0)
        ]
    }
