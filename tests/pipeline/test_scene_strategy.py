import pytest

from clipper.pipeline.scene_strategy import derive_clip_mode
from clipper.pipeline.vision import FaceBox, VisionFrame, VisionTimeline


def _face(center_x: float = 0.5, center_y: float = 0.5) -> FaceBox:
    return FaceBox(
        center_x=center_x,
        center_y=center_y,
        width=0.3,
        height=0.4,
        confidence=0.9,
    )


def _frame(timestamp: float, *, face: FaceBox | None) -> VisionFrame:
    return VisionFrame(
        timestamp_seconds=timestamp,
        motion_score=0.2,
        shot_change=False,
        primary_face=face,
    )


def test_derive_clip_mode_returns_track_when_face_present_most_frames() -> None:
    timeline = VisionTimeline(
        frames=[
            _frame(0.0, face=_face()),
            _frame(0.5, face=_face()),
            _frame(1.0, face=None),
        ]
    )

    assert derive_clip_mode(timeline, start_seconds=0.0, end_seconds=1.0) == "TRACK"


def test_derive_clip_mode_returns_general_when_faces_mostly_absent() -> None:
    timeline = VisionTimeline(
        frames=[
            _frame(0.0, face=None),
            _frame(0.5, face=None),
            _frame(1.0, face=_face()),
        ]
    )

    assert derive_clip_mode(timeline, start_seconds=0.0, end_seconds=1.0) == "GENERAL"


def test_derive_clip_mode_returns_general_without_any_frames() -> None:
    timeline = VisionTimeline(frames=[_frame(10.0, face=_face())])

    assert derive_clip_mode(timeline, start_seconds=0.0, end_seconds=1.0) == "GENERAL"


def test_derive_clip_mode_rejects_invalid_window() -> None:
    timeline = VisionTimeline(frames=[])
    with pytest.raises(ValueError):
        derive_clip_mode(timeline, start_seconds=1.0, end_seconds=1.0)


def test_derive_clip_mode_rejects_invalid_threshold() -> None:
    timeline = VisionTimeline(frames=[_frame(0.0, face=_face())])
    with pytest.raises(ValueError):
        derive_clip_mode(
            timeline, start_seconds=0.0, end_seconds=1.0, presence_threshold=1.5
        )


def test_derive_clip_mode_honors_custom_threshold() -> None:
    timeline = VisionTimeline(
        frames=[
            _frame(0.0, face=_face()),
            _frame(0.5, face=None),
            _frame(1.0, face=None),
        ]
    )

    assert (
        derive_clip_mode(
            timeline,
            start_seconds=0.0,
            end_seconds=1.0,
            presence_threshold=0.3,
        )
        == "TRACK"
    )
    assert (
        derive_clip_mode(
            timeline,
            start_seconds=0.0,
            end_seconds=1.0,
            presence_threshold=0.5,
        )
        == "GENERAL"
    )
