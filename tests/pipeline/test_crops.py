import pytest

from clipper.pipeline.crops import build_crop_plans
from clipper.pipeline.vision import FaceBox, VisionFrame, VisionTimeline


def _frame(
    timestamp: float,
    *,
    face: FaceBox | None = None,
    motion: float = 0.2,
) -> VisionFrame:
    return VisionFrame(
        timestamp_seconds=timestamp,
        motion_score=motion,
        shot_change=False,
        primary_face=face,
    )


def _face(center_x: float, center_y: float = 0.5) -> FaceBox:
    return FaceBox(
        center_x=center_x,
        center_y=center_y,
        width=0.3,
        height=0.4,
        confidence=0.95,
    )


def test_build_crop_plans_emits_one_plan_per_ratio() -> None:
    timeline = VisionTimeline(
        frames=[_frame(t / 10.0, face=_face(0.4)) for t in range(10)]
    )

    plans = build_crop_plans(
        timeline,
        start_seconds=0.0,
        end_seconds=1.0,
        source_width=1920,
        source_height=1080,
    )

    assert set(plans.keys()) == {"9:16", "1:1", "16:9"}
    for ratio, plan in plans.items():
        assert plan.aspect_ratio == ratio
        assert plan.source_width == 1920
        assert plan.source_height == 1080


def test_build_crop_plans_computes_aspect_correct_target_dims() -> None:
    timeline = VisionTimeline(frames=[_frame(0.0, face=_face(0.5))])

    plans = build_crop_plans(
        timeline,
        start_seconds=0.0,
        end_seconds=1.0,
        source_width=1920,
        source_height=1080,
    )

    assert plans["9:16"].target_width <= 1920
    assert plans["9:16"].target_height == 1080
    assert plans["9:16"].target_width % 2 == 0
    assert plans["1:1"].target_width == plans["1:1"].target_height
    assert plans["16:9"].target_width == 1920
    assert plans["16:9"].target_height == 1080


def test_build_crop_plans_anchors_track_primary_face() -> None:
    timeline = VisionTimeline(
        frames=[
            _frame(0.0, face=_face(0.3)),
            _frame(0.5, face=_face(0.4)),
            _frame(1.0, face=_face(0.5)),
        ]
    )

    plan = build_crop_plans(
        timeline,
        start_seconds=0.0,
        end_seconds=1.0,
        source_width=1920,
        source_height=1080,
        ratios=("9:16",),
    )["9:16"]

    assert len(plan.anchors) == 3
    assert plan.anchors[0].timestamp_seconds == pytest.approx(0.0)
    assert plan.anchors[2].timestamp_seconds == pytest.approx(1.0)
    assert plan.anchors[0].center_x < plan.anchors[2].center_x


def test_build_crop_plans_falls_back_to_center_without_faces() -> None:
    timeline = VisionTimeline(frames=[_frame(0.0, face=None), _frame(0.5, face=None)])

    plan = build_crop_plans(
        timeline,
        start_seconds=0.0,
        end_seconds=1.0,
        source_width=1920,
        source_height=1080,
        ratios=("9:16",),
    )["9:16"]

    for anchor in plan.anchors:
        assert anchor.center_x == pytest.approx(0.5)
        assert anchor.center_y == pytest.approx(0.5)


def test_build_crop_plans_generates_synthetic_anchors_when_no_frames_overlap() -> None:
    timeline = VisionTimeline(frames=[_frame(5.0, face=_face(0.7))])

    plan = build_crop_plans(
        timeline,
        start_seconds=0.0,
        end_seconds=1.0,
        source_width=1920,
        source_height=1080,
        ratios=("1:1",),
    )["1:1"]

    assert [anchor.center_x for anchor in plan.anchors] == [0.5, 0.5]
    assert plan.anchors[-1].timestamp_seconds == pytest.approx(1.0)


def test_build_crop_plans_rejects_invalid_bounds() -> None:
    timeline = VisionTimeline(frames=[])
    with pytest.raises(ValueError):
        build_crop_plans(
            timeline,
            start_seconds=1.0,
            end_seconds=1.0,
            source_width=1920,
            source_height=1080,
        )


def test_build_crop_plans_rejects_non_positive_source_dims() -> None:
    timeline = VisionTimeline(frames=[])
    with pytest.raises(ValueError):
        build_crop_plans(
            timeline,
            start_seconds=0.0,
            end_seconds=1.0,
            source_width=0,
            source_height=1080,
        )
