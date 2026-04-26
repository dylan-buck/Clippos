"""Tests for the ingest-side ffprobe adapter (``probe_media``).

The adapter must route through ``clippos.adapters.ffmpeg_resolver`` so
ingest works on machines that only have the vendored static-ffmpeg
fallback. Previously it shelled out to a bare ``ffprobe`` command,
which meant a machine without system ffprobe failed at the first
ingest even though the render path (already on the resolver) worked
fine.
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from clippos.adapters import ffmpeg as ingest_ffmpeg


def test_probe_media_uses_resolver_supplied_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the resolver returns the vendored ffprobe path, probe_media
    must invoke that path — not the bare string ``"ffprobe"`` (which
    would resolve via PATH and miss the vendored binary entirely)."""
    vendored_ffprobe = tmp_path / "vendored_bin" / "ffprobe"
    vendored_ffprobe.parent.mkdir()
    vendored_ffprobe.write_text("#!/usr/bin/env true")

    fake_resolver = types.ModuleType("clippos.adapters.ffmpeg_resolver")

    class _NotFound(RuntimeError):
        pass

    fake_resolver.FFmpegNotFoundError = _NotFound
    fake_resolver.resolve_ffmpeg = lambda: types.SimpleNamespace(
        ffmpeg=tmp_path / "vendored_bin" / "ffmpeg",
        ffprobe=vendored_ffprobe,
        source="vendored",
    )
    monkeypatch.setitem(
        sys.modules, "clippos.adapters.ffmpeg_resolver", fake_resolver
    )

    captured: dict[str, object] = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "codec_type": "video",
                            "width": 1920,
                            "height": 1080,
                            "avg_frame_rate": "30/1",
                        },
                        {"codec_type": "audio", "sample_rate": "48000"},
                    ],
                    "format": {"duration": "12.5"},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(ingest_ffmpeg.subprocess, "run", fake_run)

    result = ingest_ffmpeg.probe_media(tmp_path / "video.mp4")

    assert captured["command"][0] == str(vendored_ffprobe)
    assert "duration" in result["format"]


def test_probe_media_raises_clear_error_when_resolver_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the resolver can't find any ffprobe (no system, no vendored),
    surface a RuntimeError that names the engine extras escape hatch
    rather than letting a bare FileNotFoundError bubble up."""
    fake_resolver = types.ModuleType("clippos.adapters.ffmpeg_resolver")

    class _NotFound(RuntimeError):
        pass

    def boom() -> None:
        raise _NotFound("not found")

    fake_resolver.FFmpegNotFoundError = _NotFound
    fake_resolver.resolve_ffmpeg = boom
    monkeypatch.setitem(
        sys.modules, "clippos.adapters.ffmpeg_resolver", fake_resolver
    )

    with pytest.raises(RuntimeError, match="engine"):
        ingest_ffmpeg.probe_media(tmp_path / "video.mp4")
