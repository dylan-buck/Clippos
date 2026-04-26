"""Tests for the workspace-cache fingerprint helper.

The fingerprint underpins job_id derivation and the transcript / vision
cache invalidation. It MUST be:

- stable for an unchanged file at an unchanged path,
- different when the path, file size, or file mtime change, and
- safe to call on a path that doesn't exist on disk yet (URL / dry-run
  scenarios) without crashing.
"""
from __future__ import annotations

import os
from pathlib import Path

from clippos.pipeline.fingerprint import (
    FINGERPRINT_HEX_LENGTH,
    compute_video_fingerprint,
)


def test_fingerprint_is_stable_for_unchanged_file(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"hello world")

    first = compute_video_fingerprint(video)
    second = compute_video_fingerprint(video)

    assert first == second
    assert len(first) == FINGERPRINT_HEX_LENGTH


def test_fingerprint_is_stable_across_equivalent_path_spellings(
    tmp_path: Path, monkeypatch
) -> None:
    """Relative vs. absolute spellings of the same file canonicalize to
    the same path, so they must produce the same fingerprint — otherwise
    `clippos /path/to/foo.mp4` and `clippos foo.mp4` from /path/to/ would
    create separate workspaces for the same content."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"hello world")

    monkeypatch.chdir(tmp_path)

    absolute = compute_video_fingerprint(video)
    relative = compute_video_fingerprint(Path("clip.mp4"))

    assert absolute == relative


def test_fingerprint_changes_when_content_size_changes(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"hello world")
    before = compute_video_fingerprint(video)

    # Append bytes to grow the file. Even if mtime resolution were too
    # coarse to register the change, st_size is folded into the
    # fingerprint so this still re-keys the workspace.
    video.write_bytes(b"hello world plus a meaningful tail")

    after = compute_video_fingerprint(video)
    assert before != after


def test_fingerprint_changes_when_mtime_changes(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"hello world")
    before = compute_video_fingerprint(video)

    stat = video.stat()
    # Bump mtime forward by a comfortable margin so we exceed any
    # filesystem timestamp granularity (HFS+ at 1s, ext4/APFS at ns).
    new_mtime = stat.st_mtime + 7.5
    os.utime(video, (stat.st_atime, new_mtime))

    after = compute_video_fingerprint(video)
    assert before != after


def test_fingerprint_changes_when_path_changes(tmp_path: Path) -> None:
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    video_a.write_bytes(b"identical content")
    video_b.write_bytes(b"identical content")
    # Force matching mtimes so the only differing input is the path.
    stat = video_a.stat()
    os.utime(video_b, (stat.st_atime, stat.st_mtime))

    fingerprint_a = compute_video_fingerprint(video_a)
    fingerprint_b = compute_video_fingerprint(video_b)

    assert fingerprint_a != fingerprint_b


def test_fingerprint_falls_back_to_path_only_when_file_missing(
    tmp_path: Path,
) -> None:
    """During the prepare step the URL hasn't been downloaded yet, so the
    path doesn't exist on disk. The fingerprint helper must still return
    a stable string rather than crashing — the file's content will be
    picked up on a later invocation once the download completes."""
    missing = tmp_path / "not-yet-downloaded.mp4"

    fingerprint = compute_video_fingerprint(missing)

    assert isinstance(fingerprint, str)
    assert len(fingerprint) == FINGERPRINT_HEX_LENGTH
    # Stable across calls when the file still doesn't exist.
    assert fingerprint == compute_video_fingerprint(missing)
