from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from clipper.models.media import AspectRatio
from clipper.models.render import (
    CaptionLine,
    CaptionWord,
    CropAnchor,
    CropPlan,
    RenderManifest,
)

SUBTITLE_SUFFIX = ".ass"

CANONICAL_OUTPUT_DIMS: dict[AspectRatio, tuple[int, int]] = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}

NORMAL_COLOR_BGR = "00FFFFFF"
EMPHASIS_COLOR_BGR = "0000F0FF"
FONT_NAME = "Helvetica Neue"


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
    ffmpeg_binary: str = "ffmpeg",
) -> list[RenderResult]:
    if not _ffmpeg_available(ffmpeg_binary):
        raise FFmpegRenderError(
            f"{ffmpeg_binary!r} not found on PATH; install FFmpeg to render"
        )

    results: list[RenderResult] = []
    for ratio, output_path in manifest.outputs.items():
        crop_plan = manifest.crop_plans[ratio]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subtitle_root = subtitle_dir or output_path.parent
        subtitle_path = subtitle_root / f"{output_path.stem}{SUBTITLE_SUFFIX}"
        _write_ass_subtitles(
            subtitle_path,
            lines=manifest.caption_plan,
            ratio=ratio,
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
    return results


def _ffmpeg_available(ffmpeg_binary: str) -> bool:
    return shutil.which(ffmpeg_binary) is not None


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
    canonical_width, canonical_height = CANONICAL_OUTPUT_DIMS[ratio]
    crop_x, crop_y = _representative_crop_origin(crop_plan=crop_plan)
    video_filter = (
        f"crop={crop_plan.target_width}:{crop_plan.target_height}:"
        f"{crop_x}:{crop_y},"
        f"scale={canonical_width}:{canonical_height}:flags=lanczos,"
        f"ass='{_escape_for_filter(subtitle_path)}'"
    )
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
        "-vf",
        video_filter,
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


def _representative_crop_origin(*, crop_plan: CropPlan) -> tuple[int, int]:
    centroid_x, centroid_y = _centroid(crop_plan.anchors)
    max_origin_x = max(crop_plan.source_width - crop_plan.target_width, 0)
    max_origin_y = max(crop_plan.source_height - crop_plan.target_height, 0)
    origin_x = int(
        round(centroid_x * crop_plan.source_width - crop_plan.target_width / 2)
    )
    origin_y = int(
        round(centroid_y * crop_plan.source_height - crop_plan.target_height / 2)
    )
    return (
        _clamp_int(origin_x, 0, max_origin_x),
        _clamp_int(origin_y, 0, max_origin_y),
    )


def _centroid(anchors: list[CropAnchor]) -> tuple[float, float]:
    total_x = sum(anchor.center_x for anchor in anchors)
    total_y = sum(anchor.center_y for anchor in anchors)
    count = len(anchors)
    return total_x / count, total_y / count


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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical_width, canonical_height = CANONICAL_OUTPUT_DIMS[ratio]
    font_size = _font_size(canonical_height)
    margin_v = _margin_vertical(canonical_height)
    path.write_text(
        _render_ass_document(
            lines=lines,
            play_width=canonical_width,
            play_height=canonical_height,
            font_size=font_size,
            margin_v=margin_v,
        ),
        encoding="utf-8",
    )


def _render_ass_document(
    *,
    lines: list[CaptionLine],
    play_width: int,
    play_height: int,
    font_size: int,
    margin_v: int,
) -> str:
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
        f"&H00000000&,&H64000000&,-1,0,0,0,100,100,0,0,1,3,2,2,60,60,{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, "
        "Text\n"
    )
    events = "".join(_render_dialogue(line) for line in lines)
    return header + events


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


def _font_size(play_height: int) -> int:
    return max(int(round(play_height * 0.05)), 32)


def _margin_vertical(play_height: int) -> int:
    return max(int(round(play_height * 0.12)), 80)
