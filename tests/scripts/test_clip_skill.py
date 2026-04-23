import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "clip_skill.py"


def test_config_check_reports_missing_requirements(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "config-check",
            "--config",
            str(tmp_path / "missing.env"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["config_path"] == str(tmp_path / "missing.env")
    assert payload["env"]["HF_TOKEN"] is False
    assert "ffmpeg" in payload["bins"]
    assert "ffprobe" in payload["bins"]


def test_config_write_and_prepare_create_job_from_local_file(tmp_path: Path) -> None:
    config_path = tmp_path / ".env"
    output_dir = tmp_path / "out"
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake-video")

    write_result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "config-write",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--ratios",
            "9:16,1:1",
            "--max-candidates",
            "4",
            "--approve-top",
            "2",
            "--min-score",
            "0.75",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert write_result.returncode == 0

    prepare_result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "prepare",
            str(source),
            "--config",
            str(config_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert prepare_result.returncode == 0
    payload = json.loads(prepare_result.stdout)
    job_path = Path(payload["job_path"])
    job = json.loads(job_path.read_text(encoding="utf-8"))
    assert job["video_path"] == str(source.resolve())
    assert job["output_dir"] == str(output_dir)
    assert job["output_profile"]["ratios"] == ["9:16", "1:1"]
    assert job["max_candidates"] == 4
    assert payload["approve_top"] == 2
    assert payload["min_score"] == 0.75


def test_prepare_rejects_unknown_ratio(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake-video")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "prepare",
            str(source),
            "--output-dir",
            str(tmp_path / "out"),
            "--ratios",
            "4:5",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "Unsupported ratio" in result.stderr


def test_approve_marks_top_scoring_candidates(tmp_path: Path) -> None:
    manifest_path = tmp_path / "review-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "job_id": "job-1",
                "video_path": "/tmp/input.mp4",
                "candidates": [
                    _candidate("clip-a", 0.92),
                    _candidate("clip-b", 0.80),
                    _candidate("clip-c", 0.50),
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "approve",
            str(manifest_path),
            "--top",
            "2",
            "--min-score",
            "0.75",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["approved_clip_ids"] == ["clip-a", "clip-b"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [clip["approved"] for clip in manifest["candidates"]] == [
        True,
        True,
        False,
    ]


def test_approve_falls_back_to_best_clip_when_threshold_filters_all(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "review-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "job_id": "job-1",
                "video_path": "/tmp/input.mp4",
                "candidates": [
                    _candidate("clip-a", 0.60),
                    _candidate("clip-b", 0.55),
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "approve",
            str(manifest_path),
            "--top",
            "3",
            "--min-score",
            "0.90",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["approved_clip_ids"] == ["clip-a"]


def test_outputs_formats_render_report(tmp_path: Path) -> None:
    report_path = tmp_path / "render-report.json"
    report_path.write_text(
        json.dumps(
            {
                "job_id": "job-1",
                "video_path": "/tmp/input.mp4",
                "clips": [
                    {
                        "clip_id": "clip-a",
                        "outputs": {
                            "9:16": "renders/clip-a/clip-a-9x16.mp4",
                            "1:1": "renders/clip-a/clip-a-1x1.mp4",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "outputs", str(report_path)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert "clip-a" in result.stdout
    assert "9:16" in result.stdout
    assert str(tmp_path / "renders" / "clip-a" / "clip-a-9x16.mp4") in result.stdout


def _candidate(clip_id: str, score: float) -> dict:
    return {
        "clip_id": clip_id,
        "start_seconds": 0.0,
        "end_seconds": 20.0,
        "score": score,
        "reasons": ["reason"],
        "spike_categories": ["controversy"],
        "title": f"Title {clip_id}",
        "hook": f"Hook {clip_id}",
        "approved": False,
    }
