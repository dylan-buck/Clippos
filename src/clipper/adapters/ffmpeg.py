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
    command = [
        "ffprobe",
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
