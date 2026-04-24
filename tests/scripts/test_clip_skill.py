import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "clip_skill.py"


def _load_clip_skill_module():
    spec = importlib.util.spec_from_file_location("clip_skill", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_prepare_creates_unique_job_dirs_for_rapid_invocations(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake-video")
    output_dir = tmp_path / "out"

    job_paths: set[str] = set()
    for _ in range(4):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "prepare",
                str(source),
                "--output-dir",
                str(output_dir),
                "--ratios",
                "9:16",
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        job_paths.add(payload["job_path"])

    assert len(job_paths) == 4


def test_verify_downloaded_video_rejects_non_video_file(tmp_path: Path) -> None:
    if not shutil.which("ffprobe"):
        pytest.skip("ffprobe is required for this test")

    clip_skill = _load_clip_skill_module()
    bogus = tmp_path / "bogus.mp4"
    bogus.write_text("<html>not a video</html>", encoding="utf-8")

    with pytest.raises(ValueError, match="not a playable video"):
        clip_skill.verify_downloaded_video(bogus)


def test_verify_downloaded_video_requires_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_skill = _load_clip_skill_module()
    monkeypatch.setattr(clip_skill.shutil, "which", lambda _name: None)
    target = tmp_path / "video.mp4"
    target.write_bytes(b"fake")

    with pytest.raises(ValueError, match="ffprobe is required"):
        clip_skill.verify_downloaded_video(target)


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


def _package_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    scoring_request = {
        "rubric_version": "rubric-v1",
        "job_id": "job-1",
        "video_path": "/tmp/input.mp4",
        "rubric_prompt": "prompt",
        "response_schema": {},
        "clips": [
            {
                "clip_id": "clip-a",
                "clip_hash": "hash-a",
                "start_seconds": 10.0,
                "end_seconds": 40.0,
                "duration_seconds": 30.0,
                "transcript": "transcript-a",
                "speakers": ["speaker_1"],
                "mining_score": 0.7,
                "mining_signals": {
                    "hook": 0.8,
                    "keyword": 0.5,
                    "numeric": 0.1,
                    "interjection": 0.1,
                    "payoff": 0.6,
                    "question_to_answer": 0.0,
                    "motion": 0.3,
                    "shot_change": 0.2,
                    "face_presence": 0.7,
                    "speaker_interaction": 0.3,
                    "delivery_variance": 0.4,
                    "buried_lead": False,
                    "dangling_question": False,
                    "rambling_middle": False,
                    "reasons": ["strong hook"],
                    "spike_categories": ["controversy"],
                },
            }
        ],
    }
    (workspace / "scoring-request.json").write_text(
        json.dumps(scoring_request), encoding="utf-8"
    )
    review_manifest = {
        "job_id": "job-1",
        "video_path": "/tmp/input.mp4",
        "candidates": [
            {
                "clip_id": "clip-a",
                "start_seconds": 10.0,
                "end_seconds": 40.0,
                "score": 0.9,
                "reasons": ["strong hook"],
                "spike_categories": ["controversy"],
                "title": "Approved Title",
                "hook": "Approved Hook",
                "approved": True,
            }
        ],
    }
    (workspace / "review-manifest.json").write_text(
        json.dumps(review_manifest), encoding="utf-8"
    )


def test_package_prompt_writes_request_for_approved_clips(tmp_path: Path) -> None:
    workspace = tmp_path / "jobs" / "job-1"
    _package_workspace(workspace)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "package-prompt", str(workspace)],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["clip_ids"] == ["clip-a"]
    request_path = Path(payload["package_request"])
    assert request_path == workspace / "package-request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["clips"][0]["clip_hash"] == "hash-a"
    assert "package_prompt" in request
    assert request["response_schema"]["type"] == "object"


def test_package_save_fans_packs_into_render_dirs(tmp_path: Path) -> None:
    workspace = tmp_path / "jobs" / "job-1"
    _package_workspace(workspace)

    subprocess.run(
        [sys.executable, str(SCRIPT), "package-prompt", str(workspace)],
        check=True,
        capture_output=True,
        text=True,
    )
    request = json.loads(
        (workspace / "package-request.json").read_text(encoding="utf-8")
    )
    response = {
        "prompt_version": request["prompt_version"],
        "job_id": request["job_id"],
        "packs": [
            {
                "clip_id": "clip-a",
                "clip_hash": "hash-a",
                "titles": [f"Angle {i}" for i in range(5)],
                "thumbnail_texts": ["HOT TAKE", "BIG CLAIM", "LAST SHOT"],
                "social_caption": "Ready-to-paste caption body.",
                "hashtags": [
                    "#shorts",
                    "#creator",
                    "#podcast",
                    "#viral",
                    "#interview",
                ],
                "hooks": ["First hook.", "Second hook."],
            }
        ],
    }
    (workspace / "package-response.json").write_text(
        json.dumps(response), encoding="utf-8"
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "package-save", str(workspace)],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    clip_pack = workspace / "renders" / "clip-a" / "package.json"
    assert clip_pack.exists()
    assert payload["packs"][0]["pack_path"] == str(clip_pack)

    report = json.loads((workspace / "package-report.json").read_text(encoding="utf-8"))
    assert report["packs"][0]["clip_id"] == "clip-a"


def test_package_save_errors_when_response_invalid(tmp_path: Path) -> None:
    workspace = tmp_path / "jobs" / "job-1"
    _package_workspace(workspace)

    subprocess.run(
        [sys.executable, str(SCRIPT), "package-prompt", str(workspace)],
        check=True,
        capture_output=True,
        text=True,
    )
    (workspace / "package-response.json").write_text(
        json.dumps({"prompt_version": "wrong", "job_id": "job-1", "packs": []}),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "package-save", str(workspace)],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 2
    assert "prompt_version" in result.stderr or "packaging contract" in result.stderr
