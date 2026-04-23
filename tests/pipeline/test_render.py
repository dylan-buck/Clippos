from pathlib import Path

import pytest

from clipper.models.candidate import CandidateClip
from clipper.pipeline.transcribe import TranscriptSegment, TranscriptWord
from clipper.pipeline.vision import FaceBox, VisionFrame
from clipper.pipeline.render import (
    build_caption_lines,
    build_render_plan,
    choose_crop_anchor,
)


@pytest.fixture
def sample_review_candidate() -> CandidateClip:
    return CandidateClip(
        clip_id="clip-001",
        start_seconds=12.5,
        end_seconds=25.0,
        score=0.91,
        reasons=["clear hook", "speaker emphasis"],
        spike_categories=["hook", "emotion"],
        title="Hidden tradeoff",
        hook="Nobody tells you this part",
    )


@pytest.fixture
def sample_transcript_segment() -> TranscriptSegment:
    return TranscriptSegment(
        speaker="speaker_1",
        start_seconds=12.5,
        end_seconds=15.0,
        text="Nobody tells you this part",
        words=[
            TranscriptWord(
                text="Nobody",
                start_seconds=12.5,
                end_seconds=12.8,
                confidence=0.99,
            ),
            TranscriptWord(
                text="tells",
                start_seconds=12.8,
                end_seconds=13.1,
                confidence=0.98,
            ),
            TranscriptWord(
                text="you",
                start_seconds=13.1,
                end_seconds=13.2,
                confidence=0.97,
            ),
        ],
    )


def test_build_render_plan_outputs_all_ratios(sample_review_candidate) -> None:
    plan = build_render_plan(sample_review_candidate, approved=True)
    assert set(plan.outputs.keys()) == {"1:1", "16:9", "9:16"}
    assert plan.approved is True


def test_build_render_plan_uses_expected_output_filenames(sample_review_candidate) -> None:
    plan = build_render_plan(sample_review_candidate, approved=True)
    assert plan.outputs["9:16"] == Path(f"{sample_review_candidate.clip_id}-9x16.mp4")
    assert plan.outputs["1:1"] == Path(f"{sample_review_candidate.clip_id}-1x1.mp4")
    assert plan.outputs["16:9"] == Path(f"{sample_review_candidate.clip_id}-16x9.mp4")


def test_build_caption_lines_uses_segment_text_and_first_two_words(
    sample_transcript_segment,
) -> None:
    caption_lines = build_caption_lines(sample_transcript_segment)

    assert caption_lines == [
        {
            "text": "Nobody tells you this part",
            "emphasis": sample_transcript_segment.words[:2],
        }
    ]


def test_choose_crop_anchor_prefers_primary_face_coordinates() -> None:
    frame = VisionFrame(
        timestamp_seconds=1.0,
        motion_score=0.82,
        shot_change=False,
        primary_face=FaceBox(
            center_x=0.48,
            center_y=0.52,
            width=0.25,
            height=0.3,
            confidence=0.91,
        ),
    )

    assert choose_crop_anchor(frame) == (0.48, 0.52)


def test_choose_crop_anchor_defaults_to_center_without_face() -> None:
    frame = VisionFrame(
        timestamp_seconds=2.0,
        motion_score=0.41,
        shot_change=False,
        primary_face=None,
    )

    assert choose_crop_anchor(frame) == (0.5, 0.5)
