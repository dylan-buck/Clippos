from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from clippos.adapters import ffmpeg_render
from clippos.adapters import ffmpeg_resolver as fr
from clippos.adapters.ffmpeg_render import (
    CANONICAL_OUTPUT_DIMS,
    EMPHASIS_COLOR_BGR,
    FFmpegRenderError,
    FONT_NAME,
    NORMAL_COLOR_BGR,
    build_ffmpeg_command,
    render_clip,
)
from clippos.models.render import (
    CaptionLine,
    CaptionWord,
    CropAnchor,
    CropPlan,
    RenderManifest,
)


@pytest.fixture(autouse=True)
def _stub_render_ffmpeg_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that call ``render_clip(manifest)`` without an explicit
    ``ffmpeg_binary`` now go through ``ffmpeg_resolver.resolve_ffmpeg``,
    which would otherwise probe the system + try to download the vendored
    binary. We stub it to a fake ResolvedFFmpeg so existing tests keep
    exercising their subprocess + filter-availability stubs.
    """
    fake = fr.ResolvedFFmpeg(
        ffmpeg=Path("/usr/local/bin/ffmpeg"),
        ffprobe=Path("/usr/local/bin/ffprobe"),
        source=fr.SOURCE_SYSTEM,
    )
    monkeypatch.setattr(
        fr, "resolve_ffmpeg", lambda *, force_refresh=False: fake
    )
    fr.reset_cache()


def _anchor(t: float, x: float = 0.5, y: float = 0.5) -> CropAnchor:
    return CropAnchor(timestamp_seconds=t, center_x=x, center_y=y)


def _word(text: str, start: float, end: float, emphasis: bool = False) -> CaptionWord:
    return CaptionWord(
        text=text, start_seconds=start, end_seconds=end, emphasis=emphasis
    )


@pytest.fixture
def sample_manifest(tmp_path: Path) -> RenderManifest:
    source_video = tmp_path / "input.mp4"
    source_video.write_bytes(b"fake")

    crop_plan_9x16 = CropPlan(
        aspect_ratio="9:16",
        source_width=1920,
        source_height=1080,
        target_width=608,
        target_height=1080,
        anchors=[_anchor(0.0, 0.5), _anchor(1.0, 0.6)],
    )
    crop_plan_16x9 = CropPlan(
        aspect_ratio="16:9",
        source_width=1920,
        source_height=1080,
        target_width=1920,
        target_height=1080,
        anchors=[_anchor(0.0, 0.5), _anchor(1.0, 0.5)],
    )

    outputs = {
        "9:16": tmp_path / "renders" / "clip-001" / "clip-001-9x16.mp4",
        "16:9": tmp_path / "renders" / "clip-001" / "clip-001-16x9.mp4",
    }
    return RenderManifest(
        clip_id="clip-001",
        approved=True,
        source_video=source_video,
        start_seconds=2.0,
        end_seconds=5.0,
        outputs=outputs,
        crop_plans={"9:16": crop_plan_9x16, "16:9": crop_plan_16x9},
        caption_plan=[
            CaptionLine(
                start_seconds=0.0,
                end_seconds=1.5,
                text="Nobody tells you",
                words=[
                    _word("Nobody", 0.0, 0.4, emphasis=True),
                    _word("tells", 0.4, 0.8),
                    _word("you", 0.8, 1.5),
                ],
            )
        ],
    )


def test_build_ffmpeg_command_includes_core_codec_flags(
    sample_manifest: RenderManifest, tmp_path: Path
) -> None:
    crop_plan = sample_manifest.crop_plans["9:16"]
    output_path = sample_manifest.outputs["9:16"]
    subtitle_path = tmp_path / "subs.ass"

    command = build_ffmpeg_command(
        ffmpeg_binary="ffmpeg",
        manifest=sample_manifest,
        ratio="9:16",
        crop_plan=crop_plan,
        subtitle_path=subtitle_path,
        output_path=output_path,
    )

    assert command[0] == "ffmpeg"
    assert command[-1] == str(output_path)
    assert "-ss" in command
    assert command[command.index("-ss") + 1] == "2.000"
    assert "-to" in command
    assert command[command.index("-to") + 1] == "5.000"
    assert "libx264" in command
    assert "aac" in command
    assert "yuv420p" in command
    assert "+faststart" in command
    assert "-crf" in command
    assert command[command.index("-crf") + 1] == "18"


def test_build_ffmpeg_command_filter_chain_scales_to_canonical_dims(
    sample_manifest: RenderManifest, tmp_path: Path
) -> None:
    crop_plan = sample_manifest.crop_plans["9:16"]
    output_path = sample_manifest.outputs["9:16"]
    subtitle_path = tmp_path / "subs.ass"

    command = build_ffmpeg_command(
        ffmpeg_binary="ffmpeg",
        manifest=sample_manifest,
        ratio="9:16",
        crop_plan=crop_plan,
        subtitle_path=subtitle_path,
        output_path=output_path,
    )

    filter_value = command[command.index("-vf") + 1]
    canonical_width, canonical_height = CANONICAL_OUTPUT_DIMS["9:16"]
    assert f"crop={crop_plan.target_width}:{crop_plan.target_height}" in filter_value
    assert f"scale={canonical_width}:{canonical_height}" in filter_value
    assert "flags=lanczos" in filter_value
    assert "ass=filename=" in filter_value


def test_build_ffmpeg_command_uses_named_ass_filename_option_for_absolute_paths(
    sample_manifest: RenderManifest, tmp_path: Path
) -> None:
    crop_plan = sample_manifest.crop_plans["9:16"]
    subtitle_path = tmp_path / "nested" / "clip-001-9x16.ass"

    command = build_ffmpeg_command(
        ffmpeg_binary="ffmpeg",
        manifest=sample_manifest,
        ratio="9:16",
        crop_plan=crop_plan,
        subtitle_path=subtitle_path,
        output_path=tmp_path / "out.mp4",
    )

    filter_value = command[command.index("-vf") + 1]
    assert f"ass=filename='{subtitle_path}'" in filter_value


def test_build_ffmpeg_command_clamps_crop_origin_to_source_bounds(
    sample_manifest: RenderManifest, tmp_path: Path
) -> None:
    crop_plan = CropPlan(
        aspect_ratio="9:16",
        source_width=1920,
        source_height=1080,
        target_width=608,
        target_height=1080,
        anchors=[_anchor(0.0, 0.99, 0.99)],
    )
    command = build_ffmpeg_command(
        ffmpeg_binary="ffmpeg",
        manifest=sample_manifest,
        ratio="9:16",
        crop_plan=crop_plan,
        subtitle_path=tmp_path / "subs.ass",
        output_path=tmp_path / "out.mp4",
    )

    filter_value = command[command.index("-vf") + 1]
    crop_piece = filter_value.split(",", 1)[0]
    _, _, origin_x, origin_y = crop_piece.replace("crop=", "").split(":")
    assert int(origin_x) <= 1920 - 608
    assert int(origin_y) == 0


def test_render_clip_raises_when_explicit_ffmpeg_missing(
    sample_manifest: RenderManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the caller pins a specific binary, we still validate it.
    (Default `ffmpeg_binary=None` goes through the resolver which has
    its own fallback to a vendored binary, exercised separately.)"""
    monkeypatch.setattr(ffmpeg_render, "_ffmpeg_available", lambda _binary: False)

    with pytest.raises(FFmpegRenderError, match="not found on PATH"):
        render_clip(sample_manifest, ffmpeg_binary="ffmpeg")


def test_render_clip_raises_when_explicit_ffmpeg_lacks_ass_filter(
    sample_manifest: RenderManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit-binary path validates libass too — see comment above."""
    monkeypatch.setattr(ffmpeg_render, "_ffmpeg_available", lambda _binary: True)
    monkeypatch.setattr(ffmpeg_render, "_ffmpeg_filter_available", lambda *_args: False)

    with pytest.raises(FFmpegRenderError, match="ASS subtitle filter"):
        render_clip(sample_manifest, ffmpeg_binary="ffmpeg")


def test_render_clip_default_path_falls_back_to_resolver_when_system_ffmpeg_missing(
    sample_manifest: RenderManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the resolver can't find any ffmpeg + can't download the
    vendored fallback, the default code path surfaces a clean
    FFmpegRenderError instead of a raw ImportError."""
    def _explode_resolve(*, force_refresh: bool = False):
        raise fr.FFmpegNotFoundError("no ffmpeg with libass available")

    monkeypatch.setattr(fr, "resolve_ffmpeg", _explode_resolve)

    with pytest.raises(FFmpegRenderError, match="no ffmpeg with libass"):
        render_clip(sample_manifest)


def test_render_clip_skips_unapproved_manifest(
    sample_manifest: RenderManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    unapproved = sample_manifest.model_copy(update={"approved": False})

    def _explode(_binary: str) -> bool:
        raise AssertionError("unapproved manifests should not probe ffmpeg")

    monkeypatch.setattr(ffmpeg_render, "_ffmpeg_available", _explode)

    assert render_clip(unapproved) == []


def test_render_clip_raises_on_non_zero_exit(
    sample_manifest: RenderManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ffmpeg_render, "_ffmpeg_available", lambda _binary: True)
    monkeypatch.setattr(ffmpeg_render, "_ffmpeg_filter_available", lambda *_args: True)

    def fake_run(
        command: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=command, returncode=1, stdout="", stderr="boom"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(FFmpegRenderError, match="ffmpeg failed"):
        render_clip(sample_manifest)


def test_render_clip_writes_ass_sidecars_and_returns_results(
    sample_manifest: RenderManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ffmpeg_render, "_ffmpeg_available", lambda _binary: True)
    monkeypatch.setattr(ffmpeg_render, "_ffmpeg_filter_available", lambda *_args: True)
    invocations: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        invocations.append(list(command))
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    results = render_clip(sample_manifest)

    assert len(invocations) == 2
    assert {result.ratio for result in results} == {"9:16", "16:9"}
    for result in results:
        assert result.subtitle_path.exists()
        assert result.subtitle_path.suffix == ".ass"
        assert result.video_path == sample_manifest.outputs[result.ratio]


def test_render_clip_ass_document_has_expected_sections_and_emphasis(
    sample_manifest: RenderManifest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ffmpeg_render, "_ffmpeg_available", lambda _binary: True)
    monkeypatch.setattr(ffmpeg_render, "_ffmpeg_filter_available", lambda *_args: True)

    def fake_run(
        command: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    results = render_clip(sample_manifest)
    first = next(result for result in results if result.ratio == "9:16")
    document = first.subtitle_path.read_text(encoding="utf-8")

    canonical_width, canonical_height = CANONICAL_OUTPUT_DIMS["9:16"]
    assert "[Script Info]" in document
    assert "[V4+ Styles]" in document
    assert "[Events]" in document
    assert f"PlayResX: {canonical_width}" in document
    assert f"PlayResY: {canonical_height}" in document
    assert FONT_NAME in document
    assert f"&H{NORMAL_COLOR_BGR}&" in document
    assert f"&H{EMPHASIS_COLOR_BGR}&" in document
    assert "Dialogue:" in document
    assert "0:00:00.00" in document


def test_render_clip_honors_custom_subtitle_dir(
    sample_manifest: RenderManifest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ffmpeg_render, "_ffmpeg_available", lambda _binary: True)
    monkeypatch.setattr(ffmpeg_render, "_ffmpeg_filter_available", lambda *_args: True)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            args=command, returncode=0, stdout="", stderr=""
        ),
    )

    subtitle_dir = tmp_path / "subs"
    results = render_clip(sample_manifest, subtitle_dir=subtitle_dir)

    for result in results:
        assert result.subtitle_path.parent == subtitle_dir
        assert result.subtitle_path.exists()


def test_track_mode_emits_piecewise_crop_expression(
    sample_manifest: RenderManifest, tmp_path: Path
) -> None:
    crop_plan = CropPlan(
        aspect_ratio="9:16",
        source_width=1920,
        source_height=1080,
        target_width=608,
        target_height=1080,
        anchors=[
            _anchor(0.0, 0.3),
            _anchor(1.0, 0.5),
            _anchor(2.0, 0.7),
        ],
    )

    command = build_ffmpeg_command(
        ffmpeg_binary="ffmpeg",
        manifest=sample_manifest,
        ratio="9:16",
        crop_plan=crop_plan,
        subtitle_path=tmp_path / "subs.ass",
        output_path=tmp_path / "out.mp4",
    )

    filter_value = command[command.index("-vf") + 1]
    crop_piece = filter_value.split(",scale=", 1)[0]
    assert crop_piece.startswith("crop=608:1080:")
    # piecewise expression should reference t and contain escaped commas
    assert "if(lt(t\\," in crop_piece
    assert "\\," in crop_piece
    # nested: three anchors → two if() layers
    assert crop_piece.count("if(lt(t") == 2


def test_general_mode_emits_filter_complex_with_blur_and_overlay(
    sample_manifest: RenderManifest, tmp_path: Path
) -> None:
    general_manifest = sample_manifest.model_copy(update={"mode": "GENERAL"})
    output_path = general_manifest.outputs["9:16"]
    subtitle_path = tmp_path / "subs.ass"

    command = build_ffmpeg_command(
        ffmpeg_binary="ffmpeg",
        manifest=general_manifest,
        ratio="9:16",
        crop_plan=general_manifest.crop_plans["9:16"],
        subtitle_path=subtitle_path,
        output_path=output_path,
    )

    assert "-vf" not in command
    assert "-filter_complex" in command
    filter_graph = command[command.index("-filter_complex") + 1]
    canonical_width, canonical_height = CANONICAL_OUTPUT_DIMS["9:16"]
    assert "split=2" in filter_graph
    assert "boxblur=" in filter_graph
    assert (
        f"scale={canonical_width}:{canonical_height}:force_original_aspect_ratio=increase"
        in filter_graph
    )
    assert (
        f"scale={canonical_width}:{canonical_height}:force_original_aspect_ratio=decrease"
        in filter_graph
    )
    assert "overlay=(W-w)/2:(H-h)/2" in filter_graph
    assert "ass=filename='" in filter_graph
    # Output mapping explicit so the audio track survives.
    map_indices = [i for i, arg in enumerate(command) if arg == "-map"]
    assert len(map_indices) == 2
    assert command[map_indices[0] + 1] == "[out]"
    assert command[map_indices[1] + 1] == "0:a?"


def test_render_clip_general_mode_invokes_ffmpeg_and_writes_subs(
    sample_manifest: RenderManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    general_manifest = sample_manifest.model_copy(update={"mode": "GENERAL"})
    monkeypatch.setattr(ffmpeg_render, "_ffmpeg_available", lambda _binary: True)
    monkeypatch.setattr(ffmpeg_render, "_ffmpeg_filter_available", lambda *_args: True)

    invocations: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        invocations.append(list(command))
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    results = render_clip(general_manifest)

    assert len(invocations) == 2
    for command in invocations:
        assert "-filter_complex" in command
        assert "-vf" not in command
    assert {result.ratio for result in results} == {"9:16", "16:9"}
