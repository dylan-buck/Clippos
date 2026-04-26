from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from clippos.pipeline.fingerprint import compute_video_fingerprint

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "hermes_clippos.py"
CLIP_SKILL_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "clippos_skill.py"


def _load_hermes_module():
    spec = importlib.util.spec_from_file_location("hermes_clip", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_clip_skill_module():
    spec = importlib.util.spec_from_file_location("clip_skill", CLIP_SKILL_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        check=False,
        text=True,
    )


def test_preflight_reports_missing_requirements(tmp_path: Path) -> None:
    result = _run(["preflight", "--config", str(tmp_path / "missing.env")])

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["next_action"] in {"ready", "configure"}
    assert "missing" in payload
    assert "bins" in payload
    assert "ffmpeg_filters" in payload
    # HF_TOKEN is no longer a hard requirement — diarization works out of
    # the box. The token surfaces as an optional upgrade so the harness can
    # offer pyannote when the user explicitly asks for higher quality.
    assert "HF_TOKEN" not in payload["missing"]
    assert "optional_upgrades" in payload
    assert payload["optional_upgrades"]["hf_token"]["available"] is False
    assert "pyannote" in payload["optional_upgrades"]["hf_token"]["enables"]
    # engine_imports must surface in preflight so a "ready" report
    # actually means runnable end-to-end. Previously the preflight
    # cheerfully greenlit a venv that crashed at the first mine stage on
    # a missing whisperx (witnessed in the dogfood crypto-recap run).
    assert "engine_imports" in payload
    assert "missing_required" in payload["engine_imports"]


def test_is_objc_dylib_warning_matches_full_warning_block() -> None:
    """The macOS objc-runtime duplicate-class warning is a multi-line
    block emitted by cv2 + av + ffmpeg colliding. Strip all three lines
    of the block from live stderr so they don't bury real diagnostic
    signal — but recognize them precisely so we don't drop unrelated
    tracebacks that happen to indent or contain a dylib path."""
    hermes_clip = _load_hermes_module()

    # The header line
    assert hermes_clip._is_objc_dylib_warning(
        "objc[12345]: Class AVFFrameReceiver is implemented in both "
        "/opt/homebrew/Cellar/ffmpeg/lib/libavdevice.59.dylib (0x123) "
        "and /Users/x/.venv/lib/python3.12/site-packages/cv2/.dylibs/"
        "libavdevice.61.dylib (0x456). "
    )
    # The continuation line with the second dylib path
    assert hermes_clip._is_objc_dylib_warning(
        "  /Users/x/.venv/lib/python3.12/site-packages/cv2/.dylibs/"
        "libavdevice.61.3.100.dylib (0x123abc) loaded from cv2"
    )
    # The trailing "Which one is undefined" line
    assert hermes_clip._is_objc_dylib_warning(
        "One of the two will be used. Which one is undefined."
    )

    # Real diagnostic content must NOT be filtered
    assert not hermes_clip._is_objc_dylib_warning(
        "Traceback (most recent call last):"
    )
    assert not hermes_clip._is_objc_dylib_warning(
        "  File \"/Users/x/.venv/lib/python3.12/site-packages/torch/__init__.py\", line 42, in <module>"
    )
    assert not hermes_clip._is_objc_dylib_warning(
        "RuntimeError: CUDA out of memory"
    )
    assert not hermes_clip._is_objc_dylib_warning(
        "[clip] Mining candidates: probing media, transcribing..."
    )


def test_preflight_propagates_engine_misses_into_top_level_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When engine extras are not installed, preflight's `missing` list
    must include `engine:<module>` entries so the harness has a concrete
    install instruction. Stub the probe to simulate a clean dev venv."""
    hermes_clip = _load_hermes_module()
    clip_skill_module = hermes_clip.clip_skill

    fake_engine_status = {
        "ready": False,
        "interpreter": "/fake/.venv/bin/python",
        "python_version": "3.12.5",
        "required": {
            "torch": {"available": False, "error": "ModuleNotFoundError: torch"},
            "whisperx": {"available": False, "error": "ModuleNotFoundError: whisperx"},
        },
        "optional": {},
        "missing_required": ["torch", "whisperx"],
        "hint": "Engine extras not installed. Run pip install ...",
    }
    monkeypatch.setattr(
        clip_skill_module, "probe_engine_imports", lambda: fake_engine_status
    )
    # Pretend the system bins are fine so engine misses are the only signal.
    monkeypatch.setattr(
        clip_skill_module,
        "probe_render_ffmpeg",
        lambda: {"ready": True, "source": "system"},
    )
    monkeypatch.setattr(
        hermes_clip, "_which", lambda binary: f"/usr/local/bin/{binary}"
    )
    monkeypatch.setattr(
        clip_skill_module, "ffmpeg_filter_available", lambda _name: True
    )

    captured: dict[str, dict[str, object]] = {}
    monkeypatch.setattr(hermes_clip, "_emit", lambda payload: captured.setdefault("payload", payload))

    args = argparse.Namespace(config=tmp_path / "missing.env")
    rc = hermes_clip.cmd_preflight(args)

    assert rc == 0
    payload = captured["payload"]
    assert payload["ready"] is False
    assert payload["next_action"] == "configure"
    assert payload["engine_imports"] is fake_engine_status
    assert "engine:torch" in payload["missing"]
    assert "engine:whisperx" in payload["missing"]
    # System bin requirements must NOT show up — only the engine misses.
    assert "ffmpeg" not in payload["missing"]
    assert "ffprobe" not in payload["missing"]
    assert "ffmpeg-libass" not in payload["missing"]
    # The instructions field should mention the active interpreter so users
    # know which environment to install into.
    assert "/fake/.venv/bin/python" in payload["instructions"]


def test_advance_rejects_missing_source_and_workspace(tmp_path: Path) -> None:
    result = _run(["advance", "--config", str(tmp_path / ".env")])

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["next_action"] == "error"
    assert "--source" in payload["error"]


def test_advance_rejects_both_source_and_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    result = _run(
        [
            "advance",
            "--source",
            "foo.mp4",
            "--workspace",
            str(workspace),
            "--config",
            str(tmp_path / ".env"),
        ]
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["next_action"] == "error"


def test_start_new_job_persists_per_run_approval_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hermes = _load_hermes_module()
    output_dir = tmp_path / "out"
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    skill_job_dir = output_dir / "skill-jobs" / "0000-start"
    skill_job_dir.mkdir(parents=True)
    job_path = skill_job_dir / "job.json"
    job_path.write_text(
        json.dumps(
            {
                "video_path": str(source.resolve()),
                "output_dir": str(output_dir.resolve()),
                "output_profile": {"ratios": ["9:16"]},
                "max_candidates": 8,
            }
        ),
        encoding="utf-8",
    )

    def fake_run_clip_skill_json(flags, *, stage: str):
        assert stage == "prepare"
        assert "--max-candidates" in flags
        return {
            "job_path": str(job_path),
            "ratios": ["9:16"],
            "max_candidates": 8,
            "approve_top": 5,
            "min_score": 0.7,
        }

    monkeypatch.setattr(hermes, "_run_clip_skill_json", fake_run_clip_skill_json)
    args = argparse.Namespace(
        source=str(source),
        config=tmp_path / ".env",
        ratios=None,
        max_candidates=8,
        approve_top=2,
        min_score=0.83,
    )

    workspace, returned_job_path = hermes._start_new_job(args)

    assert returned_job_path == job_path
    sidecar = json.loads((workspace / "hermes-job.json").read_text(encoding="utf-8"))
    assert sidecar["job_path"] == str(job_path)
    assert sidecar["ratios"] == ["9:16"]
    assert sidecar["max_candidates"] == 8
    assert sidecar["approve_top"] == 2
    assert sidecar["min_score"] == 0.83


def test_approve_flags_reuse_resume_approval_options(tmp_path: Path) -> None:
    hermes = _load_hermes_module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    job_path = tmp_path / "job.json"
    job_path.write_text("{}", encoding="utf-8")
    hermes._write_resume_sidecar(
        workspace,
        job_path,
        approve_top=3,
        min_score=0.0,
    )
    args = argparse.Namespace(
        approve_top=None,
        min_score=None,
        config=tmp_path / ".env",
    )

    assert hermes._approve_flags(args, workspace) == [
        "--top",
        "3",
        "--min-score",
        "0.0",
    ]


def test_advance_emits_brief_handoff_when_brief_request_exists_and_unanswered(
    tmp_path: Path,
) -> None:
    """v1.1: when brief-request.json exists but neither
    brief-response.json nor brief-cache.json is present, advance must
    pause and emit a brief handoff so the harness can author the brief
    before scoring runs. Mirrors the scoring-handoff pattern."""
    output_dir = tmp_path / "out"
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    video_resolved = source.resolve()
    job_id = compute_video_fingerprint(video_resolved)
    workspace = output_dir / "jobs" / job_id
    workspace.mkdir(parents=True)
    # State setup: scoring-request and brief-request both exist; no
    # response files yet. State machine should land on needs-brief.
    (workspace / "scoring-request.json").write_text("{}", encoding="utf-8")
    (workspace / "brief-request.json").write_text(
        json.dumps({"job_id": job_id}), encoding="utf-8"
    )

    skill_job_dir = output_dir / "skill-jobs" / "0000-brief"
    skill_job_dir.mkdir(parents=True)
    job_path = skill_job_dir / "job.json"
    job_path.write_text(
        json.dumps(
            {
                "video_path": str(video_resolved),
                "output_dir": str(output_dir.resolve()),
                "output_profile": {
                    "ratios": ["9:16"],
                    "caption_preset": "hook-default",
                    "video_brief": True,
                },
                "max_candidates": 4,
            }
        ),
        encoding="utf-8",
    )

    result = _run(
        [
            "advance",
            "--workspace",
            str(workspace),
            "--config",
            str(tmp_path / ".env"),
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["next_action"] == "brief"
    assert payload["workspace"] == str(workspace.resolve())
    assert payload["handoff_request_path"].endswith("brief-request.json")
    assert payload["handoff_response_path"].endswith("brief-response.json")
    # Instructions must orient the harness on what the brief is for.
    assert "VideoBrief" in payload["instructions"]


def test_advance_emits_score_handoff_when_scoring_request_exists(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    video_resolved = source.resolve()
    job_id = compute_video_fingerprint(video_resolved)
    workspace = output_dir / "jobs" / job_id
    workspace.mkdir(parents=True)
    (workspace / "scoring-request.json").write_text("{}", encoding="utf-8")

    skill_job_dir = output_dir / "skill-jobs" / "0000-abcd"
    skill_job_dir.mkdir(parents=True)
    job_path = skill_job_dir / "job.json"
    job_path.write_text(
        json.dumps(
            {
                "video_path": str(video_resolved),
                "output_dir": str(output_dir.resolve()),
                "output_profile": {
                    "ratios": ["9:16"],
                    "caption_preset": "hook-default",
                },
                "max_candidates": 4,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = _run(
        [
            "advance",
            "--workspace",
            str(workspace),
            "--config",
            str(tmp_path / ".env"),
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["next_action"] == "score"
    assert payload["workspace"] == str(workspace.resolve())
    assert payload["handoff_request_path"].endswith("scoring-request.json")
    assert payload["handoff_response_path"].endswith("scoring-response.json")


def test_advance_rejects_workspace_without_job_when_stage_run_needed(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "out" / "jobs" / "orphan"
    workspace.mkdir(parents=True)
    # scoring-response.json without review-manifest.json puts the state
    # machine into `needs-review`, which requires the original job.json to
    # re-run the review stage. Without it, advance must fail clearly.
    (workspace / "scoring-request.json").write_text("{}", encoding="utf-8")
    (workspace / "scoring-response.json").write_text("{}", encoding="utf-8")

    result = _run(
        [
            "advance",
            "--workspace",
            str(workspace),
            "--config",
            str(tmp_path / ".env"),
        ]
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["next_action"] == "error"
    assert "job.json" in payload["error"]


def test_advance_writes_resume_sidecar_when_falling_back_to_skill_jobs(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    video_resolved = source.resolve()
    job_id = compute_video_fingerprint(video_resolved)
    workspace = output_dir / "jobs" / job_id
    workspace.mkdir(parents=True)
    (workspace / "scoring-request.json").write_text("{}", encoding="utf-8")
    (workspace / "scoring-response.json").write_text("{}", encoding="utf-8")

    skill_job_dir = output_dir / "skill-jobs" / "0000-abcd"
    skill_job_dir.mkdir(parents=True)
    job_path = skill_job_dir / "job.json"
    job_path.write_text(
        json.dumps(
            {
                "video_path": str(video_resolved),
                "output_dir": str(output_dir.resolve()),
                "output_profile": {
                    "ratios": ["9:16"],
                    "caption_preset": "hook-default",
                },
                "max_candidates": 4,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    hermes = _load_hermes_module()
    resolved = hermes._workspace_job_path(workspace.resolve())
    assert resolved == job_path
    assert (workspace / "hermes-job.json").exists()


def test_advance_score_handoff_includes_creator_patterns_when_history_exists(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    video_resolved = source.resolve()
    job_id = compute_video_fingerprint(video_resolved)
    workspace = output_dir / "jobs" / job_id
    workspace.mkdir(parents=True)
    (workspace / "scoring-request.json").write_text("{}", encoding="utf-8")

    skill_job_dir = output_dir / "skill-jobs" / "0000-abcd"
    skill_job_dir.mkdir(parents=True)
    job_path = skill_job_dir / "job.json"
    job_path.write_text(
        json.dumps(
            {
                "video_path": str(video_resolved),
                "output_dir": str(output_dir.resolve()),
                "output_profile": {
                    "ratios": ["9:16"],
                    "caption_preset": "hook-default",
                },
                "max_candidates": 4,
            }
        ),
        encoding="utf-8",
    )

    history_path = tmp_path / "history.jsonl"
    lines = []
    for index in range(12):
        lines.append(
            json.dumps(
                {
                    "job_id": f"job-long-{index}",
                    "clip_id": f"c{index}",
                    "recorded_at": "2026-04-24T00:00:00+00:00",
                    "duration_seconds": 80.0,
                    "score": 0.9,
                    "spike_categories": ["taboo"],
                    "ratios": ["9:16"],
                    "title": f"clip {index}",
                    "posted": False,
                    "notes": "",
                }
            )
        )
    for index in range(12):
        lines.append(
            json.dumps(
                {
                    "job_id": f"job-short-{index}",
                    "clip_id": f"s{index}",
                    "recorded_at": "2026-04-24T00:01:00+00:00",
                    "duration_seconds": 25.0,
                    "score": 0.6,
                    "spike_categories": ["action"],
                    "ratios": ["9:16"],
                    "title": f"clip {index}",
                    "posted": True,
                    "notes": "",
                }
            )
        )
    history_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = _run(
        [
            "advance",
            "--workspace",
            str(workspace),
            "--config",
            str(tmp_path / ".env"),
            "--history",
            str(history_path),
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["next_action"] == "score"
    assert "creator_patterns" in payload
    assert payload["creator_patterns"]["summary"]["total_clips"] == 24
    assert payload["creator_patterns"]["patterns"], "expected at least one pattern"


def test_advance_score_handoff_omits_creator_patterns_on_empty_history(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    video_resolved = source.resolve()
    job_id = compute_video_fingerprint(video_resolved)
    workspace = output_dir / "jobs" / job_id
    workspace.mkdir(parents=True)
    (workspace / "scoring-request.json").write_text("{}", encoding="utf-8")

    skill_job_dir = output_dir / "skill-jobs" / "0000-abcd"
    skill_job_dir.mkdir(parents=True)
    job_path = skill_job_dir / "job.json"
    job_path.write_text(
        json.dumps(
            {
                "video_path": str(video_resolved),
                "output_dir": str(output_dir.resolve()),
                "output_profile": {
                    "ratios": ["9:16"],
                    "caption_preset": "hook-default",
                },
                "max_candidates": 4,
            }
        ),
        encoding="utf-8",
    )

    result = _run(
        [
            "advance",
            "--workspace",
            str(workspace),
            "--config",
            str(tmp_path / ".env"),
            "--history",
            str(tmp_path / "missing-history.jsonl"),
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["next_action"] == "score"
    assert "creator_patterns" not in payload


def test_advance_emits_done_renders_when_render_report_exists(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    video_resolved = source.resolve()
    job_id = compute_video_fingerprint(video_resolved)
    workspace = output_dir / "jobs" / job_id
    workspace.mkdir(parents=True)

    (workspace / "scoring-request.json").write_text("{}", encoding="utf-8")
    (workspace / "scoring-response.json").write_text("{}", encoding="utf-8")
    (workspace / "review-manifest.json").write_text(
        json.dumps({"candidates": [{"clip_id": "c1", "approved": True}]}),
        encoding="utf-8",
    )
    render_report = {
        "clips": [
            {
                "clip_id": "c1",
                "outputs": {"9:16": "renders/c1/clip-9x16.mp4"},
            }
        ]
    }
    (workspace / "render-report.json").write_text(
        json.dumps(render_report), encoding="utf-8"
    )

    result = _run(
        [
            "advance",
            "--workspace",
            str(workspace),
            "--config",
            str(tmp_path / ".env"),
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["next_action"] == "done-renders"
    assert payload["workspace"] == str(workspace.resolve())
    assert payload["clips_dir"] == str(workspace.resolve() / "renders")
    assert "Final MP4s" in payload["summary"]
    clips = payload["clips"]
    assert len(clips) == 1
    render_path = clips[0]["renders"]["9:16"]
    assert render_path.endswith("renders/c1/clip-9x16.mp4")
    assert payload["feedback_prompt"]["clip_ids"] == ["c1"]
    assert "feedback" in payload["feedback_prompt"]["instructions"]
    assert payload["feedback_prompt"]["command_template"].endswith(
        "--kept <ids> --skipped <ids>"
    )
    assert payload["next_steps"]["package"]["recommended"] is True
    assert "--package" in payload["next_steps"]["package"]["command"]
    assert str(workspace.resolve() / "renders") in payload["instructions"]


def test_hermes_clip_feedback_pass_through_records_history(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    workspace = output_dir / "jobs" / "job-abc"
    workspace.mkdir(parents=True)
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")

    skill_job_dir = output_dir / "skill-jobs" / "0000-abcd"
    skill_job_dir.mkdir(parents=True)
    job_path = skill_job_dir / "job.json"
    job_path.write_text(
        json.dumps(
            {
                "video_path": str(source.resolve()),
                "output_dir": str(output_dir.resolve()),
                "output_profile": {
                    "ratios": ["9:16"],
                    "caption_preset": "hook-default",
                },
                "max_candidates": 4,
            }
        ),
        encoding="utf-8",
    )
    (workspace / "hermes-job.json").write_text(
        json.dumps({"job_path": str(job_path)}), encoding="utf-8"
    )
    (workspace / "review-manifest.json").write_text(
        json.dumps(
            {
                "job_id": "job-abc",
                "video_path": str(source.resolve()),
                "candidates": [
                    {
                        "clip_id": "c1",
                        "start_seconds": 0.0,
                        "end_seconds": 25.0,
                        "score": 0.9,
                        "reasons": ["hook"],
                        "spike_categories": ["taboo"],
                        "title": "T1",
                        "hook": "H1",
                        "approved": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    history_path = tmp_path / "history.jsonl"

    result = _run(
        [
            "feedback",
            str(workspace),
            "--kept",
            "c1",
            "--history",
            str(history_path),
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["next_action"] == "feedback-recorded"
    assert payload["history_rows_written"] == 1
    assert history_path.exists()


def test_advance_emits_package_handoff_when_package_request_exists(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    video_resolved = source.resolve()
    job_id = compute_video_fingerprint(video_resolved)
    workspace = output_dir / "jobs" / job_id
    workspace.mkdir(parents=True)

    for name in (
        "scoring-request.json",
        "scoring-response.json",
        "review-manifest.json",
        "render-report.json",
        "package-request.json",
    ):
        (workspace / name).write_text("{}", encoding="utf-8")
    (workspace / "review-manifest.json").write_text(
        json.dumps({"candidates": [{"clip_id": "c1", "approved": True}]}),
        encoding="utf-8",
    )

    result = _run(
        [
            "advance",
            "--workspace",
            str(workspace),
            "--config",
            str(tmp_path / ".env"),
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["next_action"] == "package"
    assert payload["handoff_request_path"].endswith("package-request.json")
    assert payload["handoff_response_path"].endswith("package-response.json")


def test_latest_workspace_delegates_to_clip_skill(tmp_path: Path) -> None:
    clip_skill = _load_clip_skill_module()
    config_path = tmp_path / ".env"
    output_dir = tmp_path / "out"
    workspace = output_dir / "jobs" / "job-a"
    workspace.mkdir(parents=True)
    clip_skill.write_env_file(config_path, {"CLIPPOS_OUTPUT_DIR": str(output_dir)})

    result = _run(["latest-workspace", "--config", str(config_path)])

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["next_action"] == "resume"
    assert payload["workspace"] == str(workspace.resolve())


def test_subprocess_env_prepends_src_to_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_subprocess_env` must prepend `<repo>/src` to PYTHONPATH so
    subprocess CLI invocations can import `clippos` even when the
    editable install in `sys.executable`'s environment is missing or
    broken. When PYTHONPATH is already set in the parent environment,
    its value must be preserved (appended after our `src` entry) so we
    don't clobber tooling that depends on it."""
    hermes = _load_hermes_module()

    monkeypatch.delenv("PYTHONPATH", raising=False)
    env = hermes._subprocess_env()
    expected_src = str(hermes.REPO_ROOT / "src")
    assert env["PYTHONPATH"] == expected_src

    monkeypatch.setenv("PYTHONPATH", "/some/other/path")
    env_with_existing = hermes._subprocess_env()
    parts = env_with_existing["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == expected_src
    assert "/some/other/path" in parts


def test_run_clip_skill_passes_env_and_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`_run_clip_skill` must set `env=` (with PYTHONPATH including
    `<repo>/src`) and `cwd=REPO_ROOT` so the spawned `clippos_skill.py`
    process can import the package even under a misconfigured
    interpreter. Without this, harness venv weirdness would manifest as
    a hard `ModuleNotFoundError: clippos` at the first subprocess
    boundary."""
    hermes = _load_hermes_module()
    captured: dict[str, object] = {}

    class FakeCompleted:
        returncode = 0
        stdout = "{}"
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeCompleted()

    monkeypatch.setattr(hermes.subprocess, "run", fake_run)
    out = hermes._run_clip_skill(["prepare"], workspace=None, stage="prepare")

    assert out == "{}"
    kwargs = captured["kwargs"]
    assert "env" in kwargs
    assert "cwd" in kwargs
    assert kwargs["cwd"] == str(hermes.REPO_ROOT)
    pythonpath = kwargs["env"]["PYTHONPATH"]
    assert str(hermes.REPO_ROOT / "src") in pythonpath.split(os.pathsep)


def test_run_cli_stage_passes_env_and_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`_run_cli_stage` spawns `python -m clippos.cli`, which is the
    most likely failure point for harness venv weirdness — `-m
    clippos.cli` fails hard with `No module named clippos` when the
    editable install is missing. Setting env + cwd defensively lets
    the source-layout PYTHONPATH provide a safety net. Must also
    preserve the live-stderr passthrough behavior (stderr=PIPE,
    bufsize=1, text=True)."""
    hermes = _load_hermes_module()
    captured: dict[str, object] = {}

    class FakeProc:
        returncode = 0
        stderr = iter([])

        def wait(self):
            return 0

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(hermes.subprocess, "Popen", fake_popen)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    job_path = tmp_path / "job.json"
    job_path.write_text("{}", encoding="utf-8")

    hermes._run_cli_stage(job_path, "mine", workspace=workspace)

    kwargs = captured["kwargs"]
    assert "env" in kwargs
    assert "cwd" in kwargs
    assert kwargs["cwd"] == str(hermes.REPO_ROOT)
    pythonpath = kwargs["env"]["PYTHONPATH"]
    assert str(hermes.REPO_ROOT / "src") in pythonpath.split(os.pathsep)
    # Must preserve live-stderr passthrough plumbing.
    assert kwargs["stderr"] == hermes.subprocess.PIPE
    assert kwargs["text"] is True
    assert kwargs["bufsize"] == 1


def test_workspace_from_job_matches_orchestrator_convention(tmp_path: Path) -> None:
    hermes = _load_hermes_module()
    job_path = tmp_path / "job.json"
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"fake")
    output_dir = tmp_path / "out"
    job_path.write_text(
        json.dumps(
            {
                "video_path": str(video_path),
                "output_dir": str(output_dir),
                "output_profile": {
                    "ratios": ["9:16"],
                    "caption_preset": "hook-default",
                },
            }
        ),
        encoding="utf-8",
    )

    workspace = hermes._workspace_from_job(job_path)
    expected_id = compute_video_fingerprint(video_path.resolve())
    assert workspace == output_dir.resolve() / "jobs" / expected_id
