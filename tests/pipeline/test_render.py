from pathlib import Path

import pytest

from clipper.models.analysis import MediaProbe
from clipper.models.candidate import CandidateClip
from clipper.pipeline.render import (
    build_render_plan,
    clip_render_dir,
    output_video_path,
    render_manifest_path,
)
from clipper.pipeline.transcribe import (
    TranscriptSegment,
    TranscriptTimeline,
    TranscriptWord,
)
from clipper.pipeline.vision import FaceBox, VisionFrame, VisionTimeline


@pytest.fixture
def sample_candidate() -> CandidateClip:
    return CandidateClip(
        clip_id="clip-001",
        start_seconds=2.0,
        end_seconds=6.0,
        score=0.91,
        reasons=["clear hook"],
        spike_categories=["emotional_confrontation"],
        title="Hidden tradeoff",
        hook="Nobody tells you this part",
    )


@pytest.fixture
def sample_transcript() -> TranscriptTimeline:
    return TranscriptTimeline(
        segments=[
            TranscriptSegment(
                speaker="speaker_1",
                start_seconds=2.0,
                end_seconds=6.0,
                text="Nobody tells you this secret tradeoff",
                words=[
                    TranscriptWord(
                        text=text,
                        start_seconds=start,
                        end_seconds=end,
                        confidence=0.95,
                    )
                    for text, start, end in [
                        ("Nobody", 2.0, 2.4),
                        ("tells", 2.4, 2.7),
                        ("you", 2.7, 2.9),
                        ("this", 2.9, 3.1),
                        ("secret", 3.1, 3.6),
                        ("tradeoff", 3.6, 4.2),
                    ]
                ],
            )
        ]
    )


@pytest.fixture
def sample_vision() -> VisionTimeline:
    return VisionTimeline(
        frames=[
            VisionFrame(
                timestamp_seconds=timestamp,
                motion_score=0.5,
                shot_change=False,
                primary_face=FaceBox(
                    center_x=0.55,
                    center_y=0.45,
                    width=0.3,
                    height=0.4,
                    confidence=0.95,
                ),
            )
            for timestamp in (2.0, 3.0, 4.0, 5.0, 6.0)
        ]
    )


@pytest.fixture
def sample_probe() -> MediaProbe:
    return MediaProbe(
        duration_seconds=120.0,
        width=1920,
        height=1080,
        fps=30.0,
        audio_sample_rate=48000,
    )


def test_build_render_plan_emits_all_three_ratios(
    sample_candidate, sample_transcript, sample_vision, sample_probe, tmp_path: Path
) -> None:
    manifest = build_render_plan(
        candidate=sample_candidate,
        source_video=tmp_path / "input.mp4",
        transcript=sample_transcript,
        vision=sample_vision,
        probe=sample_probe,
        workspace_dir=tmp_path,
    )

    assert manifest.clip_id == "clip-001"
    assert manifest.approved is True
    assert manifest.start_seconds == 2.0
    assert manifest.end_seconds == 6.0
    assert set(manifest.outputs.keys()) == {"9:16", "1:1", "16:9"}
    assert set(manifest.crop_plans.keys()) == {"9:16", "1:1", "16:9"}


def test_build_render_plan_routes_outputs_into_workspace(
    sample_candidate, sample_transcript, sample_vision, sample_probe, tmp_path: Path
) -> None:
    manifest = build_render_plan(
        candidate=sample_candidate,
        source_video=tmp_path / "input.mp4",
        transcript=sample_transcript,
        vision=sample_vision,
        probe=sample_probe,
        workspace_dir=tmp_path,
    )

    expected_dir = clip_render_dir(tmp_path, "clip-001")
    assert manifest.outputs["9:16"] == expected_dir / "clip-001-9x16.mp4"
    assert manifest.outputs["1:1"] == expected_dir / "clip-001-1x1.mp4"
    assert manifest.outputs["16:9"] == expected_dir / "clip-001-16x9.mp4"


def test_build_render_plan_caption_plan_covers_clip_window(
    sample_candidate, sample_transcript, sample_vision, sample_probe, tmp_path: Path
) -> None:
    manifest = build_render_plan(
        candidate=sample_candidate,
        source_video=tmp_path / "input.mp4",
        transcript=sample_transcript,
        vision=sample_vision,
        probe=sample_probe,
        workspace_dir=tmp_path,
    )

    assert manifest.caption_plan, "caption plan must not be empty"
    first = manifest.caption_plan[0]
    assert first.words[0].start_seconds == pytest.approx(0.0)
    emphasis_words = [
        word.text
        for line in manifest.caption_plan
        for word in line.words
        if word.emphasis
    ]
    assert "secret" in emphasis_words
    assert "tradeoff" in emphasis_words


def test_build_render_plan_crop_plans_pin_source_dims(
    sample_candidate, sample_transcript, sample_vision, sample_probe, tmp_path: Path
) -> None:
    manifest = build_render_plan(
        candidate=sample_candidate,
        source_video=tmp_path / "input.mp4",
        transcript=sample_transcript,
        vision=sample_vision,
        probe=sample_probe,
        workspace_dir=tmp_path,
    )

    for plan in manifest.crop_plans.values():
        assert plan.source_width == sample_probe.width
        assert plan.source_height == sample_probe.height
        assert plan.target_width <= plan.source_width
        assert plan.target_height <= plan.source_height


def test_render_manifest_path_matches_output_directory(tmp_path: Path) -> None:
    assert (
        render_manifest_path(tmp_path, "clip-042")
        == tmp_path / "renders" / "clip-042" / "render-manifest.json"
    )


def test_output_video_path_uses_ratio_in_filename(tmp_path: Path) -> None:
    assert (
        output_video_path(tmp_path, "clip-042", "9:16")
        == tmp_path / "renders" / "clip-042" / "clip-042-9x16.mp4"
    )


def test_build_render_plan_rejects_empty_ratios(
    sample_candidate, sample_transcript, sample_vision, sample_probe, tmp_path: Path
) -> None:
    with pytest.raises(ValueError):
        build_render_plan(
            candidate=sample_candidate,
            source_video=tmp_path / "input.mp4",
            transcript=sample_transcript,
            vision=sample_vision,
            probe=sample_probe,
            workspace_dir=tmp_path,
            ratios=(),
        )
