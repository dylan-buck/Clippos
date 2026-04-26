import json
import subprocess
from fractions import Fraction
from pathlib import Path


def normalize_probe_data(probe_data: dict) -> dict:
    required_fields = {
        "duration_seconds",
        "width",
        "height",
        "fps",
        "audio_sample_rate",
    }
    if required_fields.issubset(probe_data):
        return probe_data

    streams = probe_data.get("streams", [])
    format_data = probe_data.get("format", {})
    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        {},
    )
    audio_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        {},
    )

    return {
        "duration_seconds": float(format_data["duration"]),
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": float(Fraction(video_stream["avg_frame_rate"])),
        "audio_sample_rate": int(audio_stream["sample_rate"]),
    }


def probe_media(video_path: Path) -> dict:
    """Probe a media file via the resolved ffprobe binary.

    Routes through ``clippos.adapters.ffmpeg_resolver`` so ingest works
    on machines without a system ffprobe — the resolver falls back to
    the vendored ``static-ffmpeg`` binary when no system build is on
    PATH (or the system one lacks libass, which gates the render path).
    Previously this shelled out to a bare ``ffprobe`` command, so a user
    with only the vendored binary could pass preflight (render uses the
    resolver) but blow up here at the first ingest.
    """
    # Imported lazily so importing this module never triggers the
    # resolver's process-wide cache fill (or the static-ffmpeg download
    # on first call) just for callers that only want
    # ``normalize_probe_data``.
    from clippos.adapters.ffmpeg_resolver import (
        FFmpegNotFoundError,
        resolve_ffmpeg,
    )

    try:
        resolved = resolve_ffmpeg()
    except FFmpegNotFoundError as exc:
        raise RuntimeError(
            "Cannot probe media: no usable ffprobe found. Install engine "
            "extras (`pip install -e '.[engine]'`) for the vendored "
            "static-ffmpeg fallback, or install a system FFmpeg with "
            "libass (e.g. `brew reinstall ffmpeg` on macOS)."
        ) from exc

    command = [
        str(resolved.ffprobe),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    output = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(output.stdout)
