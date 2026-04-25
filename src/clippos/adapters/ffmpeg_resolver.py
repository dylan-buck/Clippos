"""Resolve a path to ffmpeg + ffprobe with libass subtitle support.

The render stage burns ASS-styled captions into mp4s via FFmpeg's `ass`
subtitle filter, which requires an FFmpeg build linked against libass.
That's the default Homebrew formula on macOS and standard on most Linux
distros, but not all installs include it (and the user may have a stale
build floating around in their PATH).

This resolver returns a usable ffmpeg in one of two ways:

1. **System ffmpeg, when it has libass** — preferred. No extra disk, no
   download, respects whatever build the user has in place.
2. **Vendored ffmpeg via the `static-ffmpeg` PyPI package** — a fallback
   that downloads a platform-specific static binary (~50 MB) on first
   call. Cross-platform (macOS arm64/x64, Linux x64/arm64, Windows x64),
   ffmpeg 7.0 with `--enable-libass`. No system permissions needed.

Callers should treat the returned `ResolvedFFmpeg` as authoritative —
the binary is guaranteed to support the `ass` filter. The result is
cached process-locally so repeated calls don't re-probe the system.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

ASS_FILTER_NAME = "ass"
SOURCE_SYSTEM = "system"
SOURCE_VENDORED = "vendored"


class FFmpegNotFoundError(RuntimeError):
    """Raised when no ffmpeg with libass support can be located or installed."""


@dataclass(frozen=True)
class ResolvedFFmpeg:
    """Paths to a libass-capable ffmpeg + ffprobe pair.

    `source` is "system" when we used the user's existing ffmpeg, or
    "vendored" when we fell back to the static-ffmpeg PyPI binary.
    """

    ffmpeg: Path
    ffprobe: Path
    source: str


_cache: ResolvedFFmpeg | None = None


def resolve_ffmpeg(*, force_refresh: bool = False) -> ResolvedFFmpeg:
    """Return paths to ffmpeg + ffprobe that support the `ass` filter.

    Tries the system ffmpeg first; falls back to downloading the
    static-ffmpeg vendored binary if the system version lacks libass or
    if no system ffmpeg is on PATH. Result is cached for the lifetime of
    the process — pass ``force_refresh=True`` to re-probe.
    """
    global _cache
    if _cache is not None and not force_refresh:
        return _cache

    system = _try_system_ffmpeg()
    if system is not None:
        _cache = system
        return system

    vendored = _try_vendored_ffmpeg()
    _cache = vendored
    return vendored


def probe_ffmpeg() -> ResolvedFFmpeg | None:
    """Like ``resolve_ffmpeg`` but returns ``None`` instead of raising.

    Useful for preflight surfaces that want to report the situation
    without crashing the caller. The probe still triggers the vendored
    download if needed; "no failure" includes the download succeeding.
    """
    try:
        return resolve_ffmpeg()
    except FFmpegNotFoundError:
        return None


def _try_system_ffmpeg() -> ResolvedFFmpeg | None:
    ff = shutil.which("ffmpeg")
    fp = shutil.which("ffprobe")
    if not ff or not fp:
        return None
    if not _has_filter(ff, ASS_FILTER_NAME):
        return None
    return ResolvedFFmpeg(
        ffmpeg=Path(ff),
        ffprobe=Path(fp),
        source=SOURCE_SYSTEM,
    )


def _try_vendored_ffmpeg() -> ResolvedFFmpeg:
    try:
        import static_ffmpeg
    except ImportError as exc:
        raise FFmpegNotFoundError(
            "FFmpeg with libass support is required to render captions but "
            "the system ffmpeg lacks it (or is not on PATH). Install the "
            "vendored fallback with: pip install static-ffmpeg "
            "(or upgrade the system build, e.g. `brew reinstall ffmpeg` on "
            "macOS, `apt install ffmpeg` on Debian/Ubuntu)."
        ) from exc

    # add_paths(weak=False) prepends static-ffmpeg's binaries to PATH and
    # blocks until they're downloaded on first use. Subsequent calls in
    # the same process are no-ops.
    static_ffmpeg.add_paths(weak=False)

    ff = shutil.which("ffmpeg")
    fp = shutil.which("ffprobe")
    if not ff or not fp:
        raise FFmpegNotFoundError(
            "static-ffmpeg add_paths() did not put ffmpeg/ffprobe on PATH; "
            "the vendored binary download may have failed."
        )
    if not _has_filter(ff, ASS_FILTER_NAME):
        raise FFmpegNotFoundError(
            f"vendored ffmpeg at {ff} does not support the `ass` filter; "
            "static-ffmpeg's bundled build is expected to include libass."
        )
    return ResolvedFFmpeg(
        ffmpeg=Path(ff),
        ffprobe=Path(fp),
        source=SOURCE_VENDORED,
    )


def _has_filter(binary: str | Path, filter_name: str) -> bool:
    """True when the given ffmpeg binary lists ``filter_name`` in -filters."""
    try:
        completed = subprocess.run(
            [str(binary), "-hide_banner", "-filters"],
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


def reset_cache() -> None:
    """Clear the process-local cache. Intended for tests."""
    global _cache
    _cache = None
