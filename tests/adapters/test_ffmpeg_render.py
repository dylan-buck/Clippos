from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from clipper.adapters import ffmpeg_render
from clipper.adapters.ffmpeg_render import (
    CANONICAL_OUTPUT_DIMS,
    EMPHASIS_COLOR_BGR,
    FFmpegRenderError,
    FONT_NAME,
    NORMAL_COLOR_BGR,
    build_ffmpeg_command,
    render_clip,
)
from clipper.models.render import (
    CaptionLine,
    CaptionWord,
    CropAnchor,
    CropPlan,
    RenderManifest,
)


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
    assert "ass=" in filter_value


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


def test_render_clip_raises_when_ffmpeg_missing(
    sample_manifest: RenderManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ffmpeg_render, "_ffmpeg_available", lambda _binary: False)

    with pytest.raises(FFmpegRenderError, match="not found on PATH"):
        render_clip(sample_manifest)


def test_render_clip_raises_on_non_zero_exit(
    sample_manifest: RenderManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ffmpeg_render, "_ffmpeg_available", lambda _binary: True)

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
