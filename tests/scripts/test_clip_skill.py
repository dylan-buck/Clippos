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


def test_download_video_url_uses_yt_dlp_print_to_locate_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for two bugs:
    1. yt-dlp's progress lines used to land on our stdout, corrupting the
       JSON the Hermes driver tries to parse.
    2. yt-dlp's "already downloaded" short-circuit produces no new file in
       the download dir, so the prior diff-based detection wrongly fell
       through to the urllib fallback (which then downloaded YouTube's
       HTML page as ".mp4" and crashed downstream).

    The fix uses `--print after_move:filepath` to get the destination path
    directly. This test confirms we ask for it and route stderr separately
    so user-visible progress doesn't leak onto stdout.
    """
    clip_skill = _load_clip_skill_module()
    download_dir = tmp_path / "downloads"
    download_dir.mkdir(parents=True)
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        # Simulate the cached/short-circuit case: file already exists, yt-dlp
        # would still print the path via after_move:filepath.
        existing = download_dir / "Already-Downloaded-abc.mp4"
        existing.write_bytes(b"fake")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=f"{existing}\n",
            stderr="[youtube] abc: Downloading webpage\n",
        )

    monkeypatch.setattr(clip_skill.subprocess, "run", fake_run)
    monkeypatch.setattr(clip_skill.shutil, "which", lambda _binary: "/usr/local/bin/yt-dlp")

    result = clip_skill.download_video_url(
        "https://www.youtube.com/watch?v=abc",
        download_dir,
    )

    assert result == (download_dir / "Already-Downloaded-abc.mp4").resolve()

    command = captured["command"]
    assert "--print" in command
    assert "after_move:filepath" in command
    # capture_output=True keeps yt-dlp's chatty stderr from polluting our
    # stdout; we re-emit stderr explicitly so the user still sees progress.
    assert captured["kwargs"].get("capture_output") is True
    assert captured["kwargs"].get("text") is True
    # Height cap: pulling YouTube's "best" stream gives you 4K@60 for
    # huge files that OOM the transcription/vision stages on most laptops.
    # We deliberately don't cap fps because many channels only publish
    # 1080p at 60 fps; capping fps would silently drop us to 480p.
    format_index = command.index("-f")
    format_string = command[format_index + 1]
    assert "height<=1080" in format_string
    assert "fps<=" not in format_string


def test_is_direct_cdn_url_matches_discord_and_telegram_hosts() -> None:
    clip_skill = _load_clip_skill_module()

    assert clip_skill.is_direct_cdn_url(
        "https://cdn.discordapp.com/attachments/123/456/video.mp4?ex=abc"
    )
    assert clip_skill.is_direct_cdn_url(
        "https://media.discordapp.net/attachments/1/2/x.mp4"
    )
    assert clip_skill.is_direct_cdn_url(
        "https://api.telegram.org/file/bot123:abc/videos/file.mp4"
    )


def test_is_direct_cdn_url_rejects_platform_urls_and_bare_paths() -> None:
    clip_skill = _load_clip_skill_module()

    assert not clip_skill.is_direct_cdn_url("https://youtube.com/watch?v=abc")
    assert not clip_skill.is_direct_cdn_url("https://example.com/video.mp4")
    assert not clip_skill.is_direct_cdn_url("/local/path.mp4")
    assert not clip_skill.is_direct_cdn_url("file:///tmp/foo.mp4")


def test_download_video_url_skips_yt_dlp_for_cdn_urls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_skill = _load_clip_skill_module()
    cdn_url = "https://cdn.discordapp.com/attachments/1/2/clip.mp4?ex=xyz"

    subprocess_calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        subprocess_calls.append(command)
        raise AssertionError("yt-dlp must not run for CDN URLs")

    direct_calls: list[tuple[str, Path]] = []

    def fake_direct_download(source: str, download_dir: Path) -> Path:
        direct_calls.append((source, download_dir))
        target = download_dir / "downloaded.mp4"
        target.write_bytes(b"fake")
        return target.resolve()

    monkeypatch.setattr(clip_skill.subprocess, "run", fake_run)
    monkeypatch.setattr(clip_skill, "download_direct_url", fake_direct_download)
    monkeypatch.setattr(clip_skill.shutil, "which", lambda _binary: "/usr/local/bin/yt-dlp")

    result = clip_skill.download_video_url(cdn_url, tmp_path / "downloads")

    assert subprocess_calls == []
    assert direct_calls == [(cdn_url, tmp_path / "downloads")]
    assert result == (tmp_path / "downloads" / "downloaded.mp4").resolve()


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
    # HF_TOKEN moved from `env` (where presence implied requiredness) to
    # `optional_upgrades` to reflect that it is now optional. The default
    # diarizer is the open-source SpeechBrain stack.
    assert "HF_TOKEN" not in payload.get("env", {})
    assert payload["optional_upgrades"]["hf_token"]["available"] is False
    assert "pyannote" in payload["optional_upgrades"]["hf_token"]["enables"]
    assert "ffmpeg" in payload["bins"]
    assert "ffprobe" in payload["bins"]
    assert "ass" in payload["ffmpeg_filters"]


def test_latest_workspace_plain_prints_bare_path(tmp_path: Path) -> None:
    config_path = tmp_path / ".env"
    output_dir = tmp_path / "out"
    workspace = output_dir / "jobs" / "job-only"
    workspace.mkdir(parents=True)
    clip_skill = _load_clip_skill_module()
    clip_skill.write_env_file(config_path, {"CLIPPER_OUTPUT_DIR": str(output_dir)})

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "latest-workspace",
            "--config",
            str(config_path),
            "--plain",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    # --plain must print exactly the path, no JSON wrapper. That lets shells
    # use `WORKSPACE=$(... latest-workspace --plain)` without piping through
    # a JSON parser.
    assert result.stdout.strip() == str(workspace.resolve())
    assert not result.stdout.strip().startswith("{")


def test_latest_workspace_resolves_newest_job_workspace_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / ".env"
    output_dir = tmp_path / "out"
    older = output_dir / "jobs" / "job-old"
    newer = output_dir / "jobs" / "job-new"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "review-manifest.json").write_text("{}", encoding="utf-8")
    (newer / "review-manifest.json").write_text("{}", encoding="utf-8")
    clip_skill = _load_clip_skill_module()
    clip_skill.write_env_file(config_path, {"CLIPPER_OUTPUT_DIR": str(output_dir)})

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "latest-workspace",
            "--config",
            str(config_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["workspace"] == str(newer.resolve())


def _seed_workspace_with_review_manifest(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Return (workspace, job_path, history_path) for feedback/history tests."""

    output_dir = tmp_path / "out"
    workspace = output_dir / "jobs" / "job-abc"
    workspace.mkdir(parents=True)
    history_path = tmp_path / "history.jsonl"

    skill_job_dir = output_dir / "skill-jobs" / "0000-test"
    skill_job_dir.mkdir(parents=True)
    job_path = skill_job_dir / "job.json"
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    job_path.write_text(
        json.dumps(
            {
                "video_path": str(source.resolve()),
                "output_dir": str(output_dir.resolve()),
                "output_profile": {
                    "ratios": ["9:16", "1:1"],
                    "caption_preset": "hook-default",
                },
                "max_candidates": 5,
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
                        "reasons": ["strong hook"],
                        "spike_categories": ["taboo"],
                        "title": "Title one",
                        "hook": "Hook one",
                        "approved": True,
                    },
                    {
                        "clip_id": "c2",
                        "start_seconds": 30.0,
                        "end_seconds": 95.0,
                        "score": 0.82,
                        "reasons": ["buried lead"],
                        "spike_categories": ["absurdity"],
                        "title": "Title two",
                        "hook": "Hook two",
                        "approved": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return workspace, job_path, history_path


def test_feedback_writes_log_and_appends_history(tmp_path: Path) -> None:
    workspace, _, history_path = _seed_workspace_with_review_manifest(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "feedback",
            str(workspace),
            "--kept",
            "c1",
            "--skipped",
            "c2",
            "--note",
            "c2=too long, rambling middle",
            "--history",
            str(history_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["workspace"] == str(workspace)
    assert payload["history_rows_written"] == 2

    feedback_log = json.loads(
        (workspace / "feedback-log.json").read_text(encoding="utf-8")
    )
    by_clip = {entry["clip_id"]: entry for entry in feedback_log["entries"]}
    assert by_clip["c1"]["posted"] is True
    assert by_clip["c2"]["posted"] is False
    assert "rambling middle" in by_clip["c2"]["notes"]

    history_rows = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(history_rows) == 2
    row_c1 = next(row for row in history_rows if row["clip_id"] == "c1")
    assert row_c1["posted"] is True
    assert row_c1["duration_seconds"] == 25.0
    assert row_c1["ratios"] == ["9:16", "1:1"]
    assert "taboo" in row_c1["spike_categories"]


def test_feedback_reads_stdin_json_payload(tmp_path: Path) -> None:
    workspace, _, history_path = _seed_workspace_with_review_manifest(tmp_path)
    payload = json.dumps(
        {
            "entries": [
                {"clip_id": "c1", "posted": True},
                {"clip_id": "c2", "posted": False, "notes": "too long"},
            ]
        }
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "feedback",
            str(workspace),
            "--json",
            "--history",
            str(history_path),
        ],
        input=payload,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    feedback_log = json.loads(
        (workspace / "feedback-log.json").read_text(encoding="utf-8")
    )
    by_clip = {entry["clip_id"]: entry for entry in feedback_log["entries"]}
    assert by_clip["c1"]["posted"] is True
    assert by_clip["c2"]["notes"] == "too long"


def test_feedback_rejects_unknown_clip_ids(tmp_path: Path) -> None:
    workspace, _, history_path = _seed_workspace_with_review_manifest(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "feedback",
            str(workspace),
            "--kept",
            "c99",
            "--history",
            str(history_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert "c99" in result.stderr


def test_history_summary_reports_patterns(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"

    lines: list[str] = []
    for index in range(12):
        lines.append(
            json.dumps(
                {
                    "job_id": f"job-{index}",
                    "clip_id": f"c{index}",
                    "recorded_at": "2026-04-24T00:00:00+00:00",
                    "duration_seconds": 80.0,
                    "score": 0.9,
                    "spike_categories": ["taboo"],
                    "ratios": ["16:9"],
                    "title": f"long clip {index}",
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
                    "title": f"short clip {index}",
                    "posted": True,
                    "notes": "",
                }
            )
        )
    history_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "history",
            "--summary",
            "--history",
            str(history_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["total_rows"] == 24
    assert payload["summary"]["kept"] == 12
    assert payload["summary"]["keep_rate"] == 0.5
    patterns = payload["patterns"]
    assert patterns, "expected at least one detected pattern"
    kinds = {pattern["kind"] for pattern in patterns}
    assert "length" in kinds or "score" in kinds or "ratio" in kinds


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
