from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from clippos.models.media import AspectRatio
from clippos.models.render import (
    CaptionLine,
    CaptionPreset,
    CaptionWord,
    CropAnchor,
    CropPlan,
    RenderManifest,
    SceneMode,
)
from clippos.pipeline.caption_styles import CaptionStyle, resolve_caption_style

SUBTITLE_SUFFIX = ".ass"

CANONICAL_OUTPUT_DIMS: dict[AspectRatio, tuple[int, int]] = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}

NORMAL_COLOR_BGR = "00FFFFFF"
EMPHASIS_COLOR_BGR = "0000F0FF"
FONT_NAME = "Helvetica Neue"

GENERAL_BLUR_STRENGTH = 20
GENERAL_BLUR_ITERATIONS = 1


@dataclass(frozen=True)
class RenderResult:
    ratio: AspectRatio
    video_path: Path
    subtitle_path: Path


class FFmpegRenderError(RuntimeError):
    pass


def render_clip(
    manifest: RenderManifest,
    *,
    subtitle_dir: Path | None = None,
    ffmpeg_binary: str | None = None,
) -> list[RenderResult]:
    if not manifest.approved:
        return []

    if ffmpeg_binary is None:
        # Default path: ask the resolver for a libass-capable binary. It
        # prefers the system ffmpeg when usable and falls back to the
        # vendored static-ffmpeg PyPI binary (downloads on first call).
        # That guarantees rendering works on any install — no
        # `brew install ffmpeg` required up front.
        from clippos.adapters.ffmpeg_resolver import (
            FFmpegNotFoundError,
            resolve_ffmpeg,
        )

        try:
            resolved = resolve_ffmpeg()
        except FFmpegNotFoundError as exc:
            raise FFmpegRenderError(str(exc)) from exc
        ffmpeg_binary = str(resolved.ffmpeg)
    else:
        # Caller pinned a specific binary (typically a test). Validate it
        # before doing any work.
        if not _ffmpeg_available(ffmpeg_binary):
            raise FFmpegRenderError(
                f"{ffmpeg_binary!r} not found on PATH; install FFmpeg to render"
            )
        if not _ffmpeg_filter_available(ffmpeg_binary, "ass"):
            raise FFmpegRenderError(
                f"{ffmpeg_binary!r} was found but does not provide the ASS subtitle filter; "
                "install an FFmpeg build with libass support to render captions"
            )

    caption_style = resolve_caption_style(manifest.caption_preset)
    results: list[RenderResult] = []
    for ratio, output_path in manifest.outputs.items():
        _status(f"FFmpeg: rendering {manifest.clip_id} as {ratio} -> {output_path}.")
        crop_plan = manifest.crop_plans[ratio]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subtitle_root = subtitle_dir or output_path.parent
        subtitle_path = subtitle_root / f"{output_path.stem}{SUBTITLE_SUFFIX}"
        _write_ass_subtitles(
            subtitle_path,
            lines=manifest.caption_plan,
            ratio=ratio,
            style=caption_style,
        )
        _run_ffmpeg(
            ffmpeg_binary=ffmpeg_binary,
            manifest=manifest,
            ratio=ratio,
            crop_plan=crop_plan,
            subtitle_path=subtitle_path,
            output_path=output_path,
        )
        results.append(
            RenderResult(
                ratio=ratio, video_path=output_path, subtitle_path=subtitle_path
            )
        )
        _status(f"FFmpeg: finished {manifest.clip_id} {ratio}.")
    return results


def _status(message: str) -> None:
    print(f"[clippos] {message}", file=sys.stderr, flush=True)


def _ffmpeg_available(ffmpeg_binary: str) -> bool:
    return shutil.which(ffmpeg_binary) is not None


def _ffmpeg_filter_available(ffmpeg_binary: str, filter_name: str) -> bool:
    try:
        completed = subprocess.run(
            [ffmpeg_binary, "-hide_banner", "-filters"],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return False
    if completed.returncode != 0:
        return False
    needle = f" {filter_name} "
    return any(needle in line for line in completed.stdout.splitlines())


def _run_ffmpeg(
    *,
    ffmpeg_binary: str,
    manifest: RenderManifest,
    ratio: AspectRatio,
    crop_plan: CropPlan,
    subtitle_path: Path,
    output_path: Path,
) -> None:
    command = build_ffmpeg_command(
        ffmpeg_binary=ffmpeg_binary,
        manifest=manifest,
        ratio=ratio,
        crop_plan=crop_plan,
        subtitle_path=subtitle_path,
        output_path=output_path,
    )
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise FFmpegRenderError(
            f"ffmpeg failed for {output_path} (exit {completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )


def build_ffmpeg_command(
    *,
    ffmpeg_binary: str,
    manifest: RenderManifest,
    ratio: AspectRatio,
    crop_plan: CropPlan,
    subtitle_path: Path,
    output_path: Path,
) -> list[str]:
    if manifest.mode == "GENERAL":
        return _build_general_command(
            ffmpeg_binary=ffmpeg_binary,
            manifest=manifest,
            ratio=ratio,
            subtitle_path=subtitle_path,
            output_path=output_path,
        )
    return _build_track_command(
        ffmpeg_binary=ffmpeg_binary,
        manifest=manifest,
        ratio=ratio,
        crop_plan=crop_plan,
        subtitle_path=subtitle_path,
        output_path=output_path,
    )


def _build_track_command(
    *,
    ffmpeg_binary: str,
    manifest: RenderManifest,
    ratio: AspectRatio,
    crop_plan: CropPlan,
    subtitle_path: Path,
    output_path: Path,
) -> list[str]:
    canonical_width, canonical_height = CANONICAL_OUTPUT_DIMS[ratio]
    x_keyframes, y_keyframes = _crop_origin_keyframes(crop_plan)
    x_expr = _piecewise_linear_expr(x_keyframes)
    y_expr = _piecewise_linear_expr(y_keyframes)
    video_filter = (
        f"crop={crop_plan.target_width}:{crop_plan.target_height}:"
        f"{x_expr}:{y_expr},"
        f"scale={canonical_width}:{canonical_height}:flags=lanczos,"
        f"ass=filename='{_escape_for_filter(subtitle_path)}'"
    )
    return _with_common_encode_flags(
        _common_input_flags(ffmpeg_binary, manifest),
        video_flags=["-vf", video_filter],
        output_path=output_path,
    )


def _build_general_command(
    *,
    ffmpeg_binary: str,
    manifest: RenderManifest,
    ratio: AspectRatio,
    subtitle_path: Path,
    output_path: Path,
) -> list[str]:
    canonical_width, canonical_height = CANONICAL_OUTPUT_DIMS[ratio]
    filter_complex = _build_general_filter_complex(
        canonical_width=canonical_width,
        canonical_height=canonical_height,
        subtitle_path=subtitle_path,
    )
    return _with_common_encode_flags(
        _common_input_flags(ffmpeg_binary, manifest),
        video_flags=[
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-map",
            "0:a?",
        ],
        output_path=output_path,
    )


def _build_general_filter_complex(
    *,
    canonical_width: int,
    canonical_height: int,
    subtitle_path: Path,
) -> str:
    subs = _escape_for_filter(subtitle_path)
    return (
        "[0:v]split=2[bg_src][fg_src];"
        f"[bg_src]scale={canonical_width}:{canonical_height}:"
        "force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={canonical_width}:{canonical_height},"
        f"boxblur={GENERAL_BLUR_STRENGTH}:{GENERAL_BLUR_ITERATIONS}[bg];"
        f"[fg_src]scale={canonical_width}:{canonical_height}:"
        "force_original_aspect_ratio=decrease:flags=lanczos[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2[composited];"
        f"[composited]ass=filename='{subs}'[out]"
    )


def _common_input_flags(ffmpeg_binary: str, manifest: RenderManifest) -> list[str]:
    return [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{manifest.start_seconds:.3f}",
        "-to",
        f"{manifest.end_seconds:.3f}",
        "-i",
        str(manifest.source_video),
    ]


def _with_common_encode_flags(
    prefix: list[str],
    *,
    video_flags: list[str],
    output_path: Path,
) -> list[str]:
    return [
        *prefix,
        *video_flags,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ac",
        "2",
        str(output_path),
    ]


def _crop_origin_keyframes(
    crop_plan: CropPlan,
) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
    max_origin_x = max(crop_plan.source_width - crop_plan.target_width, 0)
    max_origin_y = max(crop_plan.source_height - crop_plan.target_height, 0)
    x_keyframes: list[tuple[float, int]] = []
    y_keyframes: list[tuple[float, int]] = []
    for anchor in crop_plan.anchors:
        origin_x, origin_y = _anchor_to_origin(anchor, crop_plan)
        origin_x = _clamp_int(origin_x, 0, max_origin_x)
        origin_y = _clamp_int(origin_y, 0, max_origin_y)
        x_keyframes.append((anchor.timestamp_seconds, origin_x))
        y_keyframes.append((anchor.timestamp_seconds, origin_y))
    return x_keyframes, y_keyframes


def _anchor_to_origin(anchor: CropAnchor, crop_plan: CropPlan) -> tuple[int, int]:
    origin_x = int(
        round(anchor.center_x * crop_plan.source_width - crop_plan.target_width / 2)
    )
    origin_y = int(
        round(anchor.center_y * crop_plan.source_height - crop_plan.target_height / 2)
    )
    return origin_x, origin_y


def _piecewise_linear_expr(keyframes: list[tuple[float, int]]) -> str:
    """Emit an ffmpeg eval expression that linearly interpolates between
    ``(timestamp, value)`` keyframes. Commas are escaped with ``\\,`` so the
    result can be embedded inside an ffmpeg filter argument. If every keyframe
    carries the same value, the expression collapses to that scalar — this
    keeps tight-dedupe plans (e.g., 16:9 from 16:9 source) readable."""
    if not keyframes:
        return "0"
    if all(value == keyframes[0][1] for _, value in keyframes):
        return str(keyframes[0][1])
    tail = str(keyframes[-1][1])
    for index in range(len(keyframes) - 2, -1, -1):
        t0, v0 = keyframes[index]
        t1, v1 = keyframes[index + 1]
        segment_len = t1 - t0
        if segment_len <= 0 or v0 == v1:
            piece = str(v0)
        else:
            slope = v1 - v0
            piece = f"({v0}+({slope})*(t-{_fmt_time(t0)})/{_fmt_time(segment_len)})"
        tail = f"if(lt(t\\,{_fmt_time(t1)})\\,{piece}\\,{tail})"
    return tail


def _fmt_time(value: float) -> str:
    formatted = f"{value:.6f}".rstrip("0").rstrip(".")
    return formatted or "0"


def _clamp_int(value: int, lower: int, upper: int) -> int:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def _escape_for_filter(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _write_ass_subtitles(
    path: Path,
    *,
    lines: list[CaptionLine],
    ratio: AspectRatio,
    style: CaptionStyle,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical_width, canonical_height = CANONICAL_OUTPUT_DIMS[ratio]
    path.write_text(
        _render_ass_document(
            lines=lines,
            play_width=canonical_width,
            play_height=canonical_height,
            style=style,
        ),
        encoding="utf-8",
    )


def _render_ass_document(
    *,
    lines: list[CaptionLine],
    play_width: int,
    play_height: int,
    style: CaptionStyle,
) -> str:
    font_size = style.font_size(play_height)
    margin_v = style.margin_v(play_height)
    bold_flag = -1 if style.bold else 0
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_width}\n"
        f"PlayResY: {play_height}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding\n"
        f"Style: Default,{FONT_NAME},{font_size},&H{NORMAL_COLOR_BGR}&,&H000000FF&,"
        f"&H00000000&,&H64000000&,{bold_flag},0,0,0,100,100,0,0,1,"
        f"{_fmt_fixed(style.outline)},{_fmt_fixed(style.shadow)},"
        f"{style.alignment},{style.margin_l},{style.margin_r},{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, "
        "Text\n"
    )
    events = "".join(_render_dialogue(line) for line in lines)
    return header + events


def _fmt_fixed(value: float) -> str:
    formatted = f"{value:.2f}".rstrip("0").rstrip(".")
    return formatted or "0"


def _render_dialogue(line: CaptionLine) -> str:
    start = _format_ass_time(line.start_seconds)
    end = _format_ass_time(line.end_seconds)
    body = _render_line_body(line.words)
    return f"Dialogue: 0,{start},{end},Default,,0,0,0,,{body}\n"


def _render_line_body(words: list[CaptionWord]) -> str:
    parts: list[str] = []
    for index, word in enumerate(words):
        text = _escape_ass_text(word.text)
        if word.emphasis:
            parts.append(
                f"{{\\c&H{EMPHASIS_COLOR_BGR}&}}{text}{{\\c&H{NORMAL_COLOR_BGR}&}}"
            )
        else:
            parts.append(text)
        if index < len(words) - 1:
            parts.append(" ")
    return "".join(parts)


def _escape_ass_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _format_ass_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining = seconds - hours * 3600 - minutes * 60
    centi = int(round(remaining * 100))
    if centi >= 6000:
        centi -= 6000
        minutes += 1
    whole_seconds, centi_part = divmod(centi, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centi_part:02d}"


__all__ = [
    "CANONICAL_OUTPUT_DIMS",
    "CaptionPreset",
    "EMPHASIS_COLOR_BGR",
    "FFmpegRenderError",
    "FONT_NAME",
    "GENERAL_BLUR_STRENGTH",
    "NORMAL_COLOR_BGR",
    "RenderResult",
    "SceneMode",
    "build_ffmpeg_command",
    "render_clip",
]
