#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from clipper.models.job import ClipperJob  # noqa: E402

CONFIG_PATH = Path("~/.config/clipper-tool/.env").expanduser()
VALID_RATIOS = ("9:16", "1:1", "16:9")
DEFAULT_OUTPUT_DIR = Path("~/Documents/ClipperTool").expanduser()
DEFAULT_MAX_CANDIDATES = 12
DEFAULT_APPROVE_TOP = 3
DEFAULT_MIN_SCORE = 0.70


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Helper commands for the clipper agent skill."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_check = subparsers.add_parser("config-check")
    config_check.add_argument("--config", type=Path, default=CONFIG_PATH)
    config_check.set_defaults(func=cmd_config_check)

    config_write = subparsers.add_parser("config-write")
    config_write.add_argument("--config", type=Path, default=CONFIG_PATH)
    config_write.add_argument("--output-dir", type=Path)
    config_write.add_argument("--hf-token")
    config_write.add_argument("--ratios")
    config_write.add_argument("--max-candidates", type=int)
    config_write.add_argument("--approve-top", type=int)
    config_write.add_argument("--min-score", type=float)
    config_write.set_defaults(func=cmd_config_write)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("source")
    prepare.add_argument("--config", type=Path, default=CONFIG_PATH)
    prepare.add_argument("--output-dir", type=Path)
    prepare.add_argument("--ratios")
    prepare.add_argument("--max-candidates", type=int)
    prepare.set_defaults(func=cmd_prepare)

    approve = subparsers.add_parser("approve")
    approve.add_argument("review_manifest", type=Path)
    approve.add_argument("--top", type=int, default=DEFAULT_APPROVE_TOP)
    approve.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    approve.set_defaults(func=cmd_approve)

    outputs = subparsers.add_parser("outputs")
    outputs.add_argument("render_report", type=Path)
    outputs.set_defaults(func=cmd_outputs)

    return parser


def cmd_config_check(args: argparse.Namespace) -> int:
    config = merged_config(args.config)
    payload = {
        "config_path": str(args.config.expanduser()),
        "bins": {
            "ffmpeg": shutil.which("ffmpeg"),
            "ffprobe": shutil.which("ffprobe"),
            "yt-dlp": shutil.which("yt-dlp"),
        },
        "env": {
            "HF_TOKEN": bool(resolve_hf_token(config)),
            "CLIPPER_OUTPUT_DIR": bool(config.get("CLIPPER_OUTPUT_DIR")),
            "CLIPPER_RATIOS": bool(config.get("CLIPPER_RATIOS")),
        },
        "defaults": resolved_defaults(config),
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_config_write(args: argparse.Namespace) -> int:
    values: dict[str, str] = {}
    if args.output_dir is not None:
        values["CLIPPER_OUTPUT_DIR"] = str(args.output_dir.expanduser())
    if args.hf_token:
        values["HF_TOKEN"] = args.hf_token
    if args.ratios:
        values["CLIPPER_RATIOS"] = ",".join(parse_ratios(args.ratios))
    if args.max_candidates is not None:
        if args.max_candidates <= 0:
            raise ValueError("CLIPPER_MAX_CANDIDATES must be positive")
        values["CLIPPER_MAX_CANDIDATES"] = str(args.max_candidates)
    if args.approve_top is not None:
        if args.approve_top <= 0:
            raise ValueError("CLIPPER_APPROVE_TOP must be positive")
        values["CLIPPER_APPROVE_TOP"] = str(args.approve_top)
    if args.min_score is not None:
        if args.min_score < 0 or args.min_score > 1:
            raise ValueError("CLIPPER_MIN_SCORE must be between 0 and 1")
        values["CLIPPER_MIN_SCORE"] = f"{args.min_score:.2f}"

    existing = read_env_file(args.config)
    existing.update(values)
    write_env_file(args.config, existing)
    print(json.dumps({"config_path": str(args.config.expanduser()), "keys": sorted(values)}))
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    config = merged_config(args.config)
    output_dir = resolve_output_dir(args.output_dir, config)
    ratios = parse_ratios(args.ratios or config.get("CLIPPER_RATIOS", "all"))
    max_candidates = resolve_positive_int(
        args.max_candidates,
        config.get("CLIPPER_MAX_CANDIDATES"),
        DEFAULT_MAX_CANDIDATES,
        "max candidates",
    )
    approve_top = resolve_positive_int(
        None,
        config.get("CLIPPER_APPROVE_TOP"),
        DEFAULT_APPROVE_TOP,
        "approve top",
    )
    min_score = resolve_score(
        config.get("CLIPPER_MIN_SCORE"),
        DEFAULT_MIN_SCORE,
    )

    source_video = resolve_source(args.source, output_dir)
    job_dir = output_dir / "skill-jobs" / str(int(time.time()))
    job_dir.mkdir(parents=True, exist_ok=True)
    job_path = job_dir / "job.json"
    payload = {
        "video_path": str(source_video),
        "output_dir": str(output_dir),
        "output_profile": {
            "ratios": ratios,
            "caption_preset": "hook-default",
        },
        "max_candidates": max_candidates,
    }
    job = ClipperJob.model_validate(payload)
    job_path.write_text(json.dumps(job.model_dump(mode="json"), indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "job_path": str(job_path),
                "source_video": str(source_video),
                "output_dir": str(output_dir),
                "ratios": ratios,
                "max_candidates": max_candidates,
                "approve_top": approve_top,
                "min_score": min_score,
            },
            indent=2,
        )
    )
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    if args.top <= 0:
        raise ValueError("--top must be positive")
    if args.min_score < 0 or args.min_score > 1:
        raise ValueError("--min-score must be between 0 and 1")

    payload = json.loads(args.review_manifest.read_text(encoding="utf-8"))
    candidates = list(payload.get("candidates", []))
    if not candidates:
        raise ValueError("review manifest has no candidates")

    ranked = sorted(
        candidates,
        key=lambda clip: (-float(clip.get("score", 0)), str(clip.get("clip_id", ""))),
    )
    selected = [
        clip for clip in ranked if float(clip.get("score", 0)) >= args.min_score
    ][: args.top]
    if not selected:
        selected = ranked[:1]
    approved_ids = {clip["clip_id"] for clip in selected}

    for candidate in candidates:
        candidate["approved"] = candidate.get("clip_id") in approved_ids
    args.review_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "review_manifest": str(args.review_manifest),
                "approved_clip_ids": [clip["clip_id"] for clip in selected],
            },
            indent=2,
        )
    )
    return 0


def cmd_outputs(args: argparse.Namespace) -> int:
    report = json.loads(args.render_report.read_text(encoding="utf-8"))
    workspace_dir = args.render_report.parent
    lines = [f"Render report: {args.render_report}", ""]
    for clip in report.get("clips", []):
        lines.append(f"Clip {clip.get('clip_id')}:")
        for ratio, path in clip.get("outputs", {}).items():
            output = Path(path)
            if not output.is_absolute():
                output = workspace_dir / output
            lines.append(f"- {ratio}: {output}")
        lines.append("")
    print("\n".join(lines).rstrip())
    return 0


def read_env_file(path: Path) -> dict[str, str]:
    expanded = path.expanduser()
    if not expanded.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in expanded.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = strip_quotes(value.strip())
    return values


def write_env_file(path: Path, values: dict[str, str]) -> None:
    expanded = path.expanduser()
    expanded.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={quote_env_value(value)}" for key, value in sorted(values.items())]
    expanded.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merged_config(path: Path) -> dict[str, str]:
    config = read_env_file(path)
    for key in (
        "CLIPPER_OUTPUT_DIR",
        "CLIPPER_RATIOS",
        "CLIPPER_MAX_CANDIDATES",
        "CLIPPER_APPROVE_TOP",
        "CLIPPER_MIN_SCORE",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
    ):
        if os.environ.get(key):
            config[key] = os.environ[key]
    return config


def resolved_defaults(config: dict[str, str]) -> dict[str, object]:
    return {
        "output_dir": str(resolve_output_dir(None, config)),
        "ratios": parse_ratios(config.get("CLIPPER_RATIOS", "all")),
        "max_candidates": resolve_positive_int(
            None,
            config.get("CLIPPER_MAX_CANDIDATES"),
            DEFAULT_MAX_CANDIDATES,
            "max candidates",
        ),
        "approve_top": resolve_positive_int(
            None,
            config.get("CLIPPER_APPROVE_TOP"),
            DEFAULT_APPROVE_TOP,
            "approve top",
        ),
        "min_score": resolve_score(config.get("CLIPPER_MIN_SCORE"), DEFAULT_MIN_SCORE),
    }


def resolve_hf_token(config: dict[str, str]) -> str | None:
    return (
        config.get("HF_TOKEN")
        or config.get("HUGGING_FACE_HUB_TOKEN")
        or config.get("HUGGINGFACE_HUB_TOKEN")
    )


def parse_ratios(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(VALID_RATIOS)
    ratios = [item.strip() for item in raw.split(",") if item.strip()]
    if not ratios:
        raise ValueError("At least one ratio is required")
    unsupported = [ratio for ratio in ratios if ratio not in VALID_RATIOS]
    if unsupported:
        raise ValueError(
            "Unsupported ratio "
            f"{unsupported[0]!r}. Supported ratios: {', '.join(VALID_RATIOS)}"
        )
    return ratios


def resolve_output_dir(value: Path | None, config: dict[str, str]) -> Path:
    raw = value or Path(config.get("CLIPPER_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
    return Path(raw).expanduser().resolve()


def resolve_positive_int(
    explicit: int | None,
    configured: str | None,
    default: int,
    label: str,
) -> int:
    value = explicit if explicit is not None else int(configured or default)
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def resolve_score(configured: str | None, default: float) -> float:
    value = float(configured) if configured is not None else default
    if value < 0 or value > 1:
        raise ValueError("score must be between 0 and 1")
    return value


def resolve_source(source: str, output_dir: Path) -> Path:
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme in ("http", "https"):
        return download_video_url(source, output_dir / "downloads")
    if parsed.scheme == "file":
        path = Path(urllib.request.url2pathname(parsed.path))
    else:
        path = Path(source)
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Video file does not exist: {resolved}")
    return resolved


def download_video_url(source: str, download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("yt-dlp"):
        before = set(download_dir.iterdir())
        subprocess.run(
            [
                "yt-dlp",
                "-f",
                "bv*+ba/b",
                "--merge-output-format",
                "mp4",
                "-o",
                str(download_dir / "%(title).80s-%(id)s.%(ext)s"),
                source,
            ],
            check=True,
        )
        after = [path for path in download_dir.iterdir() if path not in before]
        if after:
            return max(after, key=lambda path: path.stat().st_mtime).resolve()
    return download_direct_url(source, download_dir)


def download_direct_url(source: str, download_dir: Path) -> Path:
    parsed = urllib.parse.urlparse(source)
    suffix = Path(parsed.path).suffix or ".mp4"
    target = download_dir / f"source-{int(time.time())}{suffix}"
    with urllib.request.urlopen(source) as response:
        target.write_bytes(response.read())
    return target.resolve()


def strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def quote_env_value(value: str) -> str:
    if not value or any(character.isspace() for character in value):
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    return value


if __name__ == "__main__":
    raise SystemExit(main())
