#!/usr/bin/env python3
"""Agent-loop helper for the clip skill (Hermes-first, works in any harness).

This script drives the clipper pipeline as a resumable state machine and
emits a single structured JSON payload on every call, so any agent harness
— Hermes, Claude Code, Codex, or a bespoke one — can advance the flow with
one tool call per step. It was designed around Hermes's single-`/clip`-
command shape but contains no Hermes-specific dependencies: every subcommand
is just Python + file I/O.

Subcommands:

- ``preflight``         — config/deps snapshot with ``next_action`` hint.
- ``advance``           — run deterministic stages until the next model
                          handoff (scoring or packaging) or ``done``.
- ``finalize``          — save packaging response + emit rendered outputs.
- ``latest-workspace``  — resolve newest job workspace from config.
- ``feedback``          — record kept/skipped outcomes into the creator
                          history (drives the self-improving profile).

The helper delegates to ``scripts/clip_skill.py`` and ``python -m clipper.cli``
rather than re-implementing orchestration, so existing tests stay authoritative.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from hashlib import sha1
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

CLIP_SKILL_SCRIPT = REPO_ROOT / "scripts" / "clip_skill.py"


def _load_clip_skill():
    spec = importlib.util.spec_from_file_location("clip_skill", CLIP_SKILL_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


clip_skill = _load_clip_skill()
from clipper.pipeline import creator_profile  # noqa: E402

CONFIG_PATH = clip_skill.CONFIG_PATH
HERMES_RESUME_FILENAME = "hermes-job.json"
MAX_CREATOR_PATTERNS = 5


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except HermesClipError as exc:
        _emit_error(exc.stage, str(exc), workspace=exc.workspace)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Agent-loop helper for the clip skill "
            "(Hermes-first, works in any harness)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", type=Path, default=CONFIG_PATH)
    preflight.set_defaults(func=cmd_preflight)

    advance = subparsers.add_parser("advance")
    advance.add_argument("--source")
    advance.add_argument("--workspace", type=Path)
    advance.add_argument("--config", type=Path, default=CONFIG_PATH)
    advance.add_argument("--ratios")
    advance.add_argument("--clips", type=int, dest="approve_top")
    advance.add_argument("--min-score", type=float, dest="min_score")
    advance.add_argument("--max-candidates", type=int)
    advance.add_argument(
        "--package",
        action="store_true",
        help="Continue past render into packaging.",
    )
    advance.add_argument(
        "--history",
        type=Path,
        default=clip_skill.DEFAULT_HISTORY_PATH,
    )
    advance.set_defaults(func=cmd_advance)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("workspace", type=Path)
    finalize.set_defaults(func=cmd_finalize)

    latest = subparsers.add_parser("latest-workspace")
    latest.add_argument("--config", type=Path, default=CONFIG_PATH)
    latest.add_argument(
        "--plain",
        action="store_true",
        help="Print just the workspace path (no JSON wrapper).",
    )
    latest.set_defaults(func=cmd_latest_workspace)

    feedback = subparsers.add_parser("feedback")
    feedback.add_argument("workspace", type=Path)
    feedback.add_argument("--kept", default="")
    feedback.add_argument("--skipped", default="")
    feedback.add_argument("--note", action="append", default=[])
    feedback.add_argument("--json", action="store_true")
    feedback.add_argument("--config", type=Path, default=CONFIG_PATH)
    feedback.add_argument(
        "--history",
        type=Path,
        default=clip_skill.DEFAULT_HISTORY_PATH,
    )
    feedback.set_defaults(func=cmd_feedback)

    return parser


# ---------- subcommand implementations ----------


def cmd_preflight(args: argparse.Namespace) -> int:
    config = clip_skill.merged_config(args.config)
    bins = {
        "ffmpeg": bool(_which("ffmpeg")),
        "ffprobe": bool(_which("ffprobe")),
        "yt-dlp": bool(_which("yt-dlp")),
    }
    ass_filter = clip_skill.ffmpeg_filter_available("ass")
    # Render goes through the resolver, which auto-falls-back to the
    # vendored static-ffmpeg binary when the system ffmpeg lacks libass.
    # If the resolver succeeds (system OR vendored), libass is no longer
    # a "missing" requirement.
    ffmpeg_render_status = clip_skill.probe_render_ffmpeg()
    # Mine + score will hard-crash without engine extras (whisperx, torch,
    # cv2, retinaface, etc.). Probe them here so a "ready" report actually
    # means runnable end-to-end, not just "system bins look ok".
    engine_status = clip_skill.probe_engine_imports()
    missing: list[str] = []
    if not bins["ffmpeg"] and not ffmpeg_render_status.get("ready"):
        missing.append("ffmpeg")
    if not bins["ffprobe"] and not ffmpeg_render_status.get("ready"):
        missing.append("ffprobe")
    if not ass_filter and not ffmpeg_render_status.get("ready"):
        missing.append("ffmpeg-libass")
    for module in engine_status.get("missing_required", []) or []:
        missing.append(f"engine:{module}")

    has_hf_token = bool(clip_skill.resolve_hf_token(config))
    ready = not missing
    next_action = "ready" if ready else "configure"
    payload: dict[str, Any] = {
        "next_action": next_action,
        "ready": ready,
        "missing": missing,
        "bins": bins,
        "ffmpeg_filters": {"ass": ass_filter},
        "ffmpeg_render": ffmpeg_render_status,
        "engine_imports": engine_status,
        "config_path": str(args.config.expanduser()),
        "defaults": clip_skill.resolved_defaults(config),
        # HF_TOKEN is no longer a hard requirement — the default diarizer is
        # the open-source SpeechBrain stack. Surface the token status as an
        # optional upgrade so the harness can offer pyannote when the user
        # asks for the highest-quality multi-speaker setup.
        "optional_upgrades": {
            "hf_token": {
                "available": has_hf_token,
                "enables": "pyannote/speaker-diarization-3.1 (set CLIPPER_DIARIZER=pyannote)",
            },
        },
    }
    if not ready:
        payload["instructions"] = (
            "Run `/clip config` to fix missing system requirements. The "
            "render stage needs FFmpeg with libass — install the engine "
            "extras (`pip install -e '.[engine]'`) and the vendored "
            "static-ffmpeg binary will be used automatically. If "
            "`engine:*` entries appear in `missing`, the active interpreter "
            f"({engine_status.get('interpreter')}) lacks engine extras — "
            "either install them in that interpreter or point CLIPPER_PYTHON "
            "at one that has them. Diarization works out of the box without "
            "HF_TOKEN; the token is only needed if the user explicitly "
            "wants the pyannote upgrade."
        )
    _emit(payload)
    return 0


def cmd_advance(args: argparse.Namespace) -> int:
    if not args.source and not args.workspace:
        raise HermesClipError(
            "advance requires --source <path|url> or --workspace <dir>",
            stage="start",
        )
    if args.source and args.workspace:
        raise HermesClipError(
            "advance accepts either --source or --workspace, not both",
            stage="start",
        )

    if args.source:
        _status(f"Preparing clip job from source: {args.source}")
        workspace, job_path = _start_new_job(args)
        _status(f"Workspace ready: {workspace}")
    else:
        workspace = _resolve_workspace(args.workspace)
        job_path = None  # resolved lazily below when a CLI stage needs it
        _status(f"Resuming clip job: {workspace}")

    def require_job_path() -> Path:
        nonlocal job_path
        if job_path is None:
            job_path = _workspace_job_path(workspace)
        return job_path

    state = _detect_state(workspace)

    if state == "needs-mine":
        _status(
            "Mining candidates: probing media, transcribing, diarizing, "
            "and analyzing vision.",
            workspace=workspace,
        )
        _run_cli_stage(require_job_path(), "mine", workspace=workspace)
        state = _detect_state(workspace)
        _status("Candidate mining complete.", workspace=workspace)

    if state == "needs-brief":
        # v1.1: pause for the harness to author the video brief before
        # scoring. Skipped when the job has output_profile.video_brief=False
        # (then mine doesn't write brief-request.json and _detect_state
        # never returns "needs-brief").
        _status("Waiting on harness video-brief handoff.", workspace=workspace)
        _emit(_brief_handoff(workspace))
        return 0

    if state == "needs-scoring":
        # v1.1: if a brief response just landed (or a cached brief is
        # available), embed it into scoring-request.json before emitting
        # the scoring handoff. The CLI brief stage is idempotent and
        # cheap, so running it unconditionally is safe — it returns
        # quickly when no brief is enabled / available.
        if (workspace / "brief-request.json").exists():
            try:
                _run_cli_stage(require_job_path(), "brief", workspace=workspace)
            except HermesClipError:
                # If the brief stage fails (response missing/invalid),
                # surface the error rather than silently scoring without
                # the brief.
                raise
        _status("Waiting on harness scoring handoff.", workspace=workspace)
        _emit(_scoring_handoff(workspace, history_path=args.history))
        return 0

    if state == "needs-review":
        _status("Building review manifest from model scores.", workspace=workspace)
        _run_cli_stage(require_job_path(), "review", workspace=workspace)
        state = _detect_state(workspace)
        _status("Review manifest ready.", workspace=workspace)

    if state == "needs-approve":
        _status("Approving the top clips for render.", workspace=workspace)
        _run_clip_skill(
            ["approve", str(_review_manifest_path(workspace))]
            + _approve_flags(args, workspace),
            workspace=workspace,
            stage="approve",
        )
        state = _detect_state(workspace)
        _status("Clip approvals saved.", workspace=workspace)

    if state == "needs-render":
        _status(
            "Rendering approved clips with captions and crop plans.",
            workspace=workspace,
        )
        _run_cli_stage(require_job_path(), "render", workspace=workspace)
        state = _detect_state(workspace)
        _status(
            f"Render complete. Clips are in {workspace / 'renders'}.",
            workspace=workspace,
        )

    if state == "rendered":
        if not args.package:
            _emit(_done_renders_payload(workspace))
            return 0
        _status("Preparing publish-pack handoff for rendered clips.", workspace=workspace)
        _run_clip_skill(
            ["package-prompt", str(workspace)],
            workspace=workspace,
            stage="package-prompt",
        )
        state = _detect_state(workspace)

    if state == "needs-packaging":
        _status("Waiting on harness packaging handoff.", workspace=workspace)
        _emit(_packaging_handoff(workspace, history_path=args.history))
        return 0

    if state == "needs-package-save":
        _status("Saving per-clip publish packs.", workspace=workspace)
        _run_clip_skill(
            ["package-save", str(workspace)],
            workspace=workspace,
            stage="package-save",
        )
        state = _detect_state(workspace)
        _status("Publish packs saved.", workspace=workspace)

    if state == "done":
        _emit(_done_packaging_payload(workspace))
        return 0

    raise HermesClipError(
        f"workspace is in an unexpected state: {state}",
        stage="advance",
        workspace=workspace,
    )


def cmd_finalize(args: argparse.Namespace) -> int:
    workspace = _resolve_workspace(args.workspace)
    state = _detect_state(workspace)
    if state == "needs-package-save":
        _run_clip_skill(
            ["package-save", str(workspace)],
            workspace=workspace,
            stage="package-save",
        )
        state = _detect_state(workspace)
    if state != "done":
        raise HermesClipError(
            f"finalize requires a package-response.json; current state: {state}",
            stage="finalize",
            workspace=workspace,
        )
    _emit(_done_packaging_payload(workspace))
    return 0


def cmd_latest_workspace(args: argparse.Namespace) -> int:
    config = clip_skill.merged_config(args.config)
    output_dir = clip_skill.resolve_output_dir(None, config)
    workspace = clip_skill.latest_workspace(output_dir)
    if workspace is None:
        raise HermesClipError(
            f"No clip job workspace found under {output_dir}",
            stage="latest-workspace",
        )
    if getattr(args, "plain", False):
        print(str(workspace))
        return 0
    _emit({"next_action": "resume", "workspace": str(workspace)})
    return 0


# ---------- state machine helpers ----------


def _detect_state(workspace: Path) -> str:
    if not (workspace / "scoring-request.json").exists():
        return "needs-mine"
    # v1.1 brief stage. Sequenced AFTER scoring-request.json exists
    # because mine writes both files in a single pass. The brief is a
    # gating handoff: harness must author brief-response.json before
    # we proceed to scoring. When the brief is disabled (no
    # brief-request.json was written), we skip these states entirely.
    if (workspace / "brief-request.json").exists():
        if not (workspace / "brief-response.json").exists() and not (
            workspace / "brief-cache.json"
        ).exists():
            return "needs-brief"
    if not (workspace / "scoring-response.json").exists():
        return "needs-scoring"
    if not (workspace / "review-manifest.json").exists():
        return "needs-review"
    manifest = _safe_read_json(workspace / "review-manifest.json")
    candidates = (manifest or {}).get("candidates") or []
    if candidates and not any(c.get("approved") for c in candidates):
        return "needs-approve"
    if not (workspace / "render-report.json").exists():
        return "needs-render"
    if (workspace / "package-report.json").exists():
        return "done"
    if not (workspace / "package-request.json").exists():
        return "rendered"
    if not (workspace / "package-response.json").exists():
        return "needs-packaging"
    return "needs-package-save"


def _brief_handoff(workspace: Path) -> dict[str, Any]:
    """v1.1 (docs/v1.1.md): one model handoff per video to author a
    pre-scoring VideoBrief. Mirrors _scoring_handoff shape so the
    harness loop pattern is the same: read request, write response,
    re-run advance.
    """
    request_path = workspace / "brief-request.json"
    response_path = workspace / "brief-response.json"
    return {
        "next_action": "brief",
        "workspace": str(workspace),
        "handoff_request_path": str(request_path),
        "handoff_response_path": str(response_path),
        "instructions": (
            "Read the brief request, follow its embedded `brief_prompt` "
            "and `response_schema`, author a one-paragraph VideoBrief "
            "synthesizing the video's spine + expected viral patterns + "
            "anti-patterns, then write the response file and rerun "
            "`advance`. The brief is one handoff per video and is cached "
            "for the rest of this workspace's lifetime — keep it tight "
            "and opinionated. The per-clip scorer reads this brief and "
            "uses it to bias scores toward on-thesis moments."
        ),
    }


def _scoring_handoff(workspace: Path, *, history_path: Path) -> dict[str, Any]:
    request_path = workspace / "scoring-request.json"
    response_path = workspace / "scoring-response.json"
    payload: dict[str, Any] = {
        "next_action": "score",
        "workspace": str(workspace),
        "handoff_request_path": str(request_path),
        "handoff_response_path": str(response_path),
        "instructions": (
            "Read the request file, follow its embedded `rubric_prompt` and "
            "`response_schema`, score every clip preserving `clip_id` and "
            "`clip_hash`, then write the response file and rerun `advance`. "
            "Treat any `creator_patterns` and `harness_memory` cues in this "
            "payload as contextual lens — the rubric stays authoritative."
        ),
    }
    _attach_creator_profile(payload, history_path=history_path)
    return payload


def _packaging_handoff(workspace: Path, *, history_path: Path) -> dict[str, Any]:
    request_path = workspace / "package-request.json"
    response_path = workspace / "package-response.json"
    payload: dict[str, Any] = {
        "next_action": "package",
        "workspace": str(workspace),
        "handoff_request_path": str(request_path),
        "handoff_response_path": str(response_path),
        "instructions": (
            "Read the package request, follow its embedded `package_prompt` "
            "and `response_schema`, write one PublishPack per clip preserving "
            "`clip_id` and `clip_hash`, save it as the response file, then "
            "rerun `advance` to persist the packs. Use any `creator_patterns` "
            "in this payload to shape phrasing within the schema."
        ),
    }
    _attach_creator_profile(payload, history_path=history_path)
    return payload


def _done_renders_payload(workspace: Path) -> dict[str, Any]:
    clips = _collect_clip_outputs(workspace)
    clips_dir = workspace / "renders"
    return {
        "next_action": "done-renders",
        "workspace": str(workspace),
        "clips_dir": str(clips_dir),
        "clips": clips,
        "summary": (
            f"Rendered {len(clips)} clip(s). Final MP4s are under {clips_dir}."
        ),
        "feedback_prompt": {
            "instructions": (
                "After the user reports which clips they posted vs. skipped, "
                "record it with `hermes_clip.py feedback "
                f"{workspace} --kept <ids> --skipped <ids>`. This appends to "
                "the creator history so future runs score more like the "
                "user's taste. Also offer to save any stated preferences "
                "(caption style, length, banned phrases) to harness memory."
            ),
            "clip_ids": [clip.get("clip_id") for clip in clips if clip.get("clip_id")],
        },
        "instructions": (
            f"Share the rendered clip paths with the user and mention that "
            f"all clips live in {clips_dir}. Run `advance --package` on this "
            "workspace to generate publish packs. Ask the user which clips "
            "they actually posted and record the answer via the `feedback` "
            "subcommand."
        ),
    }


def _done_packaging_payload(workspace: Path) -> dict[str, Any]:
    clips = _collect_clip_outputs(workspace, include_packages=True)
    clips_dir = workspace / "renders"
    return {
        "next_action": "done-package",
        "workspace": str(workspace),
        "clips_dir": str(clips_dir),
        "clips": clips,
        "package_report": str(workspace / "package-report.json"),
        "summary": f"Rendered clips and publish packs are under {clips_dir}.",
        "feedback_prompt": {
            "instructions": (
                "Offer the user one last chance to record which clips they "
                "will post via `hermes_clip.py feedback`. Feedback feeds the "
                "creator profile used on the next `/clip` run."
            ),
            "clip_ids": [clip.get("clip_id") for clip in clips if clip.get("clip_id")],
        },
    }


def _attach_creator_profile(payload: dict[str, Any], *, history_path: Path) -> None:
    """Enrich a handoff payload with aggregated creator-profile cues.

    Scoring/packaging are the two moments the harness model actually writes
    an output. Handing it the user's recent feedback summary + top-confidence
    patterns turns raw rubric work into taste-aware work.
    """
    try:
        raw_entries = creator_profile.load_history(history_path)
    except OSError:
        return
    if not raw_entries:
        return
    entries = creator_profile.latest_entries_by_clip(raw_entries)
    summary = creator_profile.summarize(entries)
    if summary["total_clips"] == 0:
        return
    patterns = creator_profile.detect_patterns(entries)
    payload["creator_patterns"] = {
        "summary": summary,
        "patterns": [pattern.to_json() for pattern in patterns[:MAX_CREATOR_PATTERNS]],
    }


def cmd_feedback(args: argparse.Namespace) -> int:
    flags = [
        "feedback",
        str(args.workspace),
        "--config",
        str(args.config),
        "--history",
        str(args.history),
    ]
    if args.kept:
        flags += ["--kept", args.kept]
    if args.skipped:
        flags += ["--skipped", args.skipped]
    for note in args.note or []:
        flags += ["--note", note]
    stdin_payload: str | None = None
    if args.json:
        flags.append("--json")
        stdin_payload = sys.stdin.read()
    completed = subprocess.run(
        [sys.executable, str(CLIP_SKILL_SCRIPT), *flags],
        input=stdin_payload,
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise HermesClipError(
            completed.stderr.strip() or completed.stdout.strip() or "feedback failed",
            stage="feedback",
            workspace=args.workspace,
        )
    payload = json.loads(completed.stdout)
    payload["next_action"] = "feedback-recorded"
    _emit(payload)
    return 0


def _collect_clip_outputs(
    workspace: Path, *, include_packages: bool = False
) -> list[dict[str, Any]]:
    report = _safe_read_json(workspace / "render-report.json")
    if not report:
        return []
    clips: list[dict[str, Any]] = []
    for entry in report.get("clips", []):
        clip_id = entry.get("clip_id")
        outputs: dict[str, str] = {}
        for ratio, rel in (entry.get("outputs") or {}).items():
            path = Path(rel)
            outputs[ratio] = str(path if path.is_absolute() else workspace / path)
        clip: dict[str, Any] = {"clip_id": clip_id, "renders": outputs}
        if include_packages and clip_id:
            package_path = workspace / "renders" / clip_id / "package.json"
            if package_path.exists():
                clip["package"] = str(package_path)
        clips.append(clip)
    return clips


# ---------- job bootstrapping ----------


def _start_new_job(args: argparse.Namespace) -> tuple[Path, Path]:
    flags = ["prepare", args.source, "--config", str(args.config)]
    if args.ratios:
        flags += ["--ratios", args.ratios]
    if args.max_candidates is not None:
        flags += ["--max-candidates", str(args.max_candidates)]
    payload = _run_clip_skill_json(flags, stage="prepare")
    job_path = Path(payload["job_path"])
    workspace = _workspace_from_job(job_path)
    workspace.mkdir(parents=True, exist_ok=True)
    approve_top = (
        args.approve_top
        if args.approve_top is not None
        else _coerce_positive_int(payload.get("approve_top"))
    )
    min_score = (
        args.min_score
        if args.min_score is not None
        else _coerce_score(payload.get("min_score"))
    )
    ratios = _coerce_ratios(payload.get("ratios"))
    max_candidates = _coerce_positive_int(payload.get("max_candidates"))
    _write_resume_sidecar(
        workspace,
        job_path,
        ratios=ratios,
        max_candidates=max_candidates,
        approve_top=approve_top,
        min_score=min_score,
    )
    return workspace, job_path


def _workspace_from_job(job_path: Path) -> Path:
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    output_dir = Path(payload["output_dir"]).expanduser().resolve()
    video_path = Path(payload["video_path"]).expanduser().resolve(strict=False)
    job_id = sha1(str(video_path).encode()).hexdigest()[:12]
    return output_dir / "jobs" / job_id


def _read_resume_sidecar(workspace: Path) -> dict[str, Any]:
    sidecar = workspace / HERMES_RESUME_FILENAME
    if not sidecar.exists():
        return {}
    payload = _safe_read_json(sidecar) or {}
    return payload if isinstance(payload, dict) else {}


def _write_resume_sidecar(
    workspace: Path,
    job_path: Path,
    *,
    ratios: list[str] | None = None,
    max_candidates: int | None = None,
    approve_top: int | None = None,
    min_score: float | None = None,
) -> None:
    sidecar = workspace / HERMES_RESUME_FILENAME
    payload: dict[str, Any] = {"job_path": str(job_path)}
    if ratios is not None:
        payload["ratios"] = ratios
    if max_candidates is not None:
        payload["max_candidates"] = max_candidates
    if approve_top is not None:
        payload["approve_top"] = approve_top
    if min_score is not None:
        payload["min_score"] = min_score
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _workspace_job_path(workspace: Path) -> Path:
    payload = _read_resume_sidecar(workspace)
    job_path = payload.get("job_path")
    if job_path and Path(job_path).exists():
        return Path(job_path)
    fallback = _find_matching_skill_job(workspace)
    if fallback is not None:
        _write_resume_sidecar(workspace, fallback)
        return fallback
    raise HermesClipError(
        f"Cannot resolve job.json for workspace {workspace}. Pass --source to "
        "start a fresh run, or restore hermes-job.json pointing to the original "
        "job.json.",
        stage="advance",
        workspace=workspace,
    )


def _find_matching_skill_job(workspace: Path) -> Path | None:
    output_root = workspace.parent.parent
    skill_jobs = output_root / "skill-jobs"
    if not skill_jobs.exists():
        return None
    for candidate in skill_jobs.glob("*/job.json"):
        try:
            expected = _workspace_from_job(candidate)
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        if expected == workspace:
            return candidate
    return None


# ---------- subprocess helpers ----------


def _run_clip_skill_json(
    flags: list[str], *, stage: str, workspace: Path | None = None
) -> dict[str, Any]:
    stdout = _run_clip_skill(flags, workspace=workspace, stage=stage)
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise HermesClipError(
            f"clip_skill {stage} produced invalid JSON: {exc}",
            stage=stage,
            workspace=workspace,
        ) from exc


def _run_clip_skill(
    flags: list[str], *, workspace: Path | None, stage: str
) -> str:
    completed = subprocess.run(
        [sys.executable, str(CLIP_SKILL_SCRIPT), *flags],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or (
            f"clip_skill {stage} exited {completed.returncode}"
        )
        raise HermesClipError(message, stage=stage, workspace=workspace)
    return completed.stdout


def _run_cli_stage(job_path: Path, stage: str, *, workspace: Path) -> None:
    """Run a clipper.cli stage with live stderr passthrough.

    Mining + render take 5-30 minutes on real videos. A blanket
    ``capture_output=True`` would swallow all of WhisperX, SpeechBrain,
    and RAFT's progress until the very end, so users have no signal that
    anything is happening — and silent OOM kills look identical to a
    healthy long-running job.

    We stream the child's stderr live to our own stderr so model
    downloads, transcription progress, and crashes are all visible in
    real time. We tail the last 60 stderr lines into a ring buffer so
    error messages stay informative when a stage exits non-zero.
    Stdout is discarded — the CLI prints the artifact path there but we
    already know it from the workspace layout.

    macOS prints a wall of `objc[xxxxx]: Class X is implemented in both
    libfoo.dylib and libbar.dylib — One of the two will be used. Which one
    is undefined.` warnings on every run because cv2, av (pyannote dep),
    static-ffmpeg, and the system Homebrew ffmpeg each bundle their own
    copy of libav*. They are cosmetic — no functional bug — but they
    pollute user-visible stderr and bury real errors. Filter them out of
    the live passthrough; preserve them in the ring buffer so they are
    available if a stage exits non-zero (rare; the warnings rarely
    correlate with actual failures, but keeping them in the error tail
    avoids hiding diagnostic signal).
    """
    from collections import deque

    _status(f"Starting stage `{stage}`.", workspace=workspace)
    proc = subprocess.Popen(
        [sys.executable, "-m", "clipper.cli", "run", str(job_path), "--stage", stage],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    tail: deque[str] = deque(maxlen=60)
    assert proc.stderr is not None
    for line in proc.stderr:
        tail.append(line)
        if not _is_objc_dylib_warning(line):
            sys.stderr.write(line)
            sys.stderr.flush()
    proc.wait()
    if proc.returncode != 0:
        message = (
            "".join(tail).strip()
            or f"clipper stage {stage} exited {proc.returncode}"
        )
        raise HermesClipError(message, stage=stage, workspace=workspace)
    _status(f"Finished stage `{stage}`.", workspace=workspace)


# Pattern matches the macOS Objective-C runtime's "Class … is implemented
# in both …" duplicate-class warnings emitted by every cv2 / av / ffmpeg
# import on a system with multiple libav copies. These are cosmetic; we
# strip them from the live stderr passthrough so they don't bury real
# diagnostic signal.
_OBJC_DUPLICATE_CLASS_WARNING = ("objc[", "is implemented in both")


def _is_objc_dylib_warning(line: str) -> bool:
    """Return True for the multi-line objc dylib duplicate-class warning.

    The warning shape is:

        objc[12345]: Class FooBar is implemented in both /a/lib1.dylib (...)
        and /b/lib2.dylib (...). One of the two will be used. Which one is undefined.

    Matches both the leading `objc[NNNN]: Class ... implemented in both`
    line and the continuation lines that name the dylibs and the
    "One of the two will be used" disclaimer.
    """
    stripped = line.lstrip()
    if stripped.startswith(_OBJC_DUPLICATE_CLASS_WARNING[0]) and (
        _OBJC_DUPLICATE_CLASS_WARNING[1] in stripped
    ):
        return True
    if "One of the two will be used. Which one is undefined." in stripped:
        return True
    # The continuation lines (paths to the dylibs) start with whitespace
    # followed by `/`. Match conservatively so we don't drop unrelated
    # tracebacks that also indent.
    if stripped.startswith("/") and (
        ".dylib" in stripped or ".framework/" in stripped
    ) and ("(0x" in stripped or "loaded from" in stripped):
        return True
    return False


# ---------- misc helpers ----------


def _resolve_workspace(workspace: Path) -> Path:
    resolved = workspace.expanduser().resolve()
    if not resolved.exists():
        raise HermesClipError(
            f"workspace does not exist: {resolved}",
            stage="advance",
            workspace=resolved,
        )
    return resolved


def _review_manifest_path(workspace: Path) -> Path:
    path = workspace / "review-manifest.json"
    if not path.exists():
        raise HermesClipError(
            f"{workspace} is missing review-manifest.json",
            stage="approve",
            workspace=workspace,
        )
    return path


def _approve_flags(args: argparse.Namespace, workspace: Path) -> list[str]:
    config = clip_skill.merged_config(args.config)
    defaults = clip_skill.resolved_defaults(config)
    resume = _read_resume_sidecar(workspace)
    resume_top = _coerce_positive_int(resume.get("approve_top"))
    resume_min_score = _coerce_score(resume.get("min_score"))
    top = (
        args.approve_top
        if args.approve_top is not None
        else resume_top if resume_top is not None else defaults["approve_top"]
    )
    min_score = (
        args.min_score
        if args.min_score is not None
        else resume_min_score if resume_min_score is not None else defaults["min_score"]
    )
    return ["--top", str(top), "--min-score", str(min_score)]


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _coerce_score(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if 0 <= parsed <= 1 else None


def _coerce_ratios(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    ratios = [item for item in value if isinstance(item, str)]
    return ratios or None


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _which(binary: str) -> str | None:
    import shutil

    return shutil.which(binary)


def _status(message: str, *, workspace: Path | None = None) -> None:
    prefix = "[clip]"
    if workspace is not None:
        prefix += f" [{workspace.name}]"
    print(f"{prefix} {message}", file=sys.stderr, flush=True)


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2))


def _emit_error(stage: str, message: str, *, workspace: Path | None) -> None:
    payload: dict[str, Any] = {
        "next_action": "error",
        "stage": stage,
        "error": message,
    }
    if workspace is not None:
        payload["workspace"] = str(workspace)
    _emit(payload)


class HermesClipError(RuntimeError):
    def __init__(
        self, message: str, *, stage: str, workspace: Path | None = None
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.workspace = workspace


if __name__ == "__main__":
    raise SystemExit(main())
