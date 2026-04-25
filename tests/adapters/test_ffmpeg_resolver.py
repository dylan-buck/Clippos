"""Tests for the libass-aware ffmpeg resolver.

The resolver decides between the user's system ffmpeg and a vendored
static-ffmpeg binary. Tests stub `shutil.which`, the `ass`-filter probe,
and the `static_ffmpeg` import to exercise the four interesting branches:

1. System ffmpeg has libass → use it, skip download.
2. System ffmpeg lacks libass → fall back to vendored.
3. No system ffmpeg + vendored available → use vendored.
4. No system ffmpeg + no static_ffmpeg installed → raise cleanly.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from clippos.adapters import ffmpeg_resolver as fr


@pytest.fixture(autouse=True)
def _reset_resolver_cache() -> None:
    """Each test starts with a clean resolver cache so prior outcomes
    don't leak into subsequent ones."""
    fr.reset_cache()
    yield
    fr.reset_cache()


def test_resolve_ffmpeg_prefers_system_when_libass_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The system path is the no-disk, no-download happy case."""
    monkeypatch.setattr(
        fr.shutil,
        "which",
        lambda name: f"/usr/local/bin/{name}" if name in ("ffmpeg", "ffprobe") else None,
    )
    monkeypatch.setattr(fr, "_has_filter", lambda _binary, _filter: True)

    resolved = fr.resolve_ffmpeg()

    assert resolved.source == fr.SOURCE_SYSTEM
    assert resolved.ffmpeg == Path("/usr/local/bin/ffmpeg")
    assert resolved.ffprobe == Path("/usr/local/bin/ffprobe")


def test_resolve_ffmpeg_falls_back_to_vendored_when_system_lacks_libass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The "you have ffmpeg but it can't burn captions" case — the most
    important fallback. We stub static_ffmpeg.add_paths to drop a binary
    into a fake bin dir and make shutil.which return that path on the
    second call."""
    vendored_dir = tmp_path / "static_ffmpeg" / "darwin_arm64"
    vendored_dir.mkdir(parents=True)
    vendored_ff = vendored_dir / "ffmpeg"
    vendored_ff.write_text("#!/usr/bin/env true")
    vendored_fp = vendored_dir / "ffprobe"
    vendored_fp.write_text("#!/usr/bin/env true")

    # Phase 1: system has ffmpeg/ffprobe but no libass.
    # Phase 2: after add_paths(), shutil.which finds the vendored binary.
    state = {"vendored_active": False}

    def fake_which(name: str) -> str | None:
        if state["vendored_active"]:
            return str(vendored_dir / name) if name in ("ffmpeg", "ffprobe") else None
        return f"/usr/local/bin/{name}" if name in ("ffmpeg", "ffprobe") else None

    def fake_has_filter(binary: str | Path, _filter: str) -> bool:
        # System binary lacks libass; vendored has it.
        return str(binary).startswith(str(vendored_dir))

    fake_static_ffmpeg = types.ModuleType("static_ffmpeg")

    def fake_add_paths(*, weak: bool) -> None:
        state["vendored_active"] = True

    fake_static_ffmpeg.add_paths = fake_add_paths  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "static_ffmpeg", fake_static_ffmpeg)
    monkeypatch.setattr(fr.shutil, "which", fake_which)
    monkeypatch.setattr(fr, "_has_filter", fake_has_filter)

    resolved = fr.resolve_ffmpeg()

    assert resolved.source == fr.SOURCE_VENDORED
    assert resolved.ffmpeg == vendored_ff
    assert resolved.ffprobe == vendored_fp


def test_resolve_ffmpeg_uses_vendored_when_no_system_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No system ffmpeg at all — the vendored binary is the only option."""
    vendored_dir = tmp_path / "vendored"
    vendored_dir.mkdir(parents=True)
    (vendored_dir / "ffmpeg").write_text("")
    (vendored_dir / "ffprobe").write_text("")

    state = {"vendored_active": False}

    def fake_which(name: str) -> str | None:
        if not state["vendored_active"]:
            return None
        return str(vendored_dir / name) if name in ("ffmpeg", "ffprobe") else None

    fake_static_ffmpeg = types.ModuleType("static_ffmpeg")

    def fake_add_paths(*, weak: bool) -> None:
        state["vendored_active"] = True

    fake_static_ffmpeg.add_paths = fake_add_paths  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "static_ffmpeg", fake_static_ffmpeg)
    monkeypatch.setattr(fr.shutil, "which", fake_which)
    monkeypatch.setattr(fr, "_has_filter", lambda *_args: True)

    resolved = fr.resolve_ffmpeg()

    assert resolved.source == fr.SOURCE_VENDORED


def test_resolve_ffmpeg_raises_clean_error_when_static_ffmpeg_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When neither the system nor the vendored fallback is available,
    the error message must point users at how to fix it."""
    monkeypatch.setattr(fr.shutil, "which", lambda _name: None)
    monkeypatch.setattr(fr, "_has_filter", lambda *_args: False)
    # Make sure the import fails. Removing from sys.modules is enough
    # because the resolver imports lazily.
    monkeypatch.setitem(sys.modules, "static_ffmpeg", None)

    with pytest.raises(fr.FFmpegNotFoundError) as excinfo:
        fr.resolve_ffmpeg()

    message = str(excinfo.value)
    # Must point users at the fix. Both engine extras and system install
    # are valid solutions; the message mentions both.
    assert "static-ffmpeg" in message
    assert "libass" in message


def test_probe_ffmpeg_returns_none_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """probe_ffmpeg is the preflight-friendly variant — preflight UIs
    should report status, not crash."""
    monkeypatch.setattr(fr.shutil, "which", lambda _name: None)
    monkeypatch.setattr(fr, "_has_filter", lambda *_args: False)
    monkeypatch.setitem(sys.modules, "static_ffmpeg", None)

    assert fr.probe_ffmpeg() is None


def test_resolve_ffmpeg_caches_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated calls in the same process should not re-probe — `-filters`
    invocations are slow enough to matter when called on a hot path."""
    call_count = {"has_filter": 0}

    def fake_has_filter(_binary: Any, _filter: str) -> bool:
        call_count["has_filter"] += 1
        return True

    monkeypatch.setattr(
        fr.shutil,
        "which",
        lambda name: f"/usr/local/bin/{name}" if name in ("ffmpeg", "ffprobe") else None,
    )
    monkeypatch.setattr(fr, "_has_filter", fake_has_filter)

    first = fr.resolve_ffmpeg()
    second = fr.resolve_ffmpeg()

    assert first is second  # cached object identity
    assert call_count["has_filter"] == 1


def test_resolve_ffmpeg_force_refresh_re_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """force_refresh=True bypasses the cache."""
    monkeypatch.setattr(
        fr.shutil,
        "which",
        lambda name: f"/usr/local/bin/{name}" if name in ("ffmpeg", "ffprobe") else None,
    )
    monkeypatch.setattr(fr, "_has_filter", lambda *_args: True)

    first = fr.resolve_ffmpeg()
    second = fr.resolve_ffmpeg(force_refresh=True)

    assert first.source == second.source == fr.SOURCE_SYSTEM
    assert first is not second
