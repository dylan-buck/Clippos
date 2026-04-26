"""Stable workspace-cache fingerprint for source videos.

The job_id and adapter caches (transcript.json, vision.json) used to be
keyed only by the canonical video path. That meant editing/re-encoding a
video at the same path silently reused the previous transcript and
vision artifacts, leading to "why is it clipping the wrong thing?"
correctness bugs when the cached payloads no longer matched the current
content.

The fingerprint folds together canonical path + ``st_size`` +
``st_mtime_ns`` + ``clippos.__version__``. Any change to the file's
size or mtime forces a fresh job_id (new workspace) and invalidates the
adapter caches. Bumping the package version invalidates all caches,
which is the safe default when the contract or model changes.

We deliberately do NOT include a full content hash: hashing a 2 GB
podcast on every run defeats the cache. mtime+size catches the common
case (re-export, re-download, manual edit). If a paranoid workflow ever
needs byte-exact invalidation, fold a partial-content SHA-256 in here.
"""
from __future__ import annotations

from hashlib import sha1
from pathlib import Path

from clippos import __version__ as _CLIPPOS_VERSION

FINGERPRINT_HEX_LENGTH = 12


def canonical_video_path(video_path: Path) -> Path:
    """Return the canonical absolute path used everywhere as the cache key root."""
    return video_path.expanduser().resolve(strict=False)


def compute_video_fingerprint(video_path: Path) -> str:
    """Compute the workspace fingerprint for a source video path.

    Stable for an unchanged file at an unchanged path under an unchanged
    clippos version. Changes when the file's size, mtime, path, or the
    clippos version changes.

    For sources that don't have a stat-able file on disk yet (URLs that
    the prepare step is about to download, missing paths during dry
    runs), fall back to a path-only fingerprint so callers don't crash.
    Once the file exists on a subsequent invocation the fingerprint will
    naturally pick up size+mtime and re-key the workspace.
    """
    canonical = canonical_video_path(video_path)
    parts: list[str] = [str(canonical), _CLIPPOS_VERSION]
    try:
        stat = canonical.stat()
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
        # No stat available — degrade to path-only fingerprint. This is
        # the same behavior the old sha1(path) derivation had, so we
        # don't regress any existing flows that key on a not-yet-fetched
        # source.
        pass
    else:
        parts.append(str(stat.st_size))
        parts.append(str(stat.st_mtime_ns))

    digest = sha1("\0".join(parts).encode("utf-8")).hexdigest()
    return digest[:FINGERPRINT_HEX_LENGTH]


__all__ = [
    "FINGERPRINT_HEX_LENGTH",
    "canonical_video_path",
    "compute_video_fingerprint",
]
