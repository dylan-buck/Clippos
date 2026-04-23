from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipper.adapters import vision as vision_adapter
from clipper.adapters.vision import (
    DEFAULT_MODEL,
    FrameSample,
    RawFace,
    VisionConfig,
    VisionError,
    _extract_faces,
    analyze,
    build_frames,
    normalize_motion_scores,
    select_primary_face,
    smooth_face_trajectory,
)
from clipper.pipeline.vision import build_vision_timeline


def test_vision_config_defaults_match_quality_first_choices() -> None:
    config = VisionConfig()

    assert config.sample_fps == 2.0
    assert config.scene_threshold == 27.0
    assert config.face_min_confidence == 0.5
    assert config.motion_frame_width == 256
    assert config.one_euro_min_cutoff == 1.0
    assert config.one_euro_beta == 0.1


def test_select_primary_face_returns_none_when_empty() -> None:
    assert select_primary_face([]) is None


def test_select_primary_face_picks_highest_area_weighted_confidence() -> None:
    small = RawFace(center_x=0.1, center_y=0.1, width=0.1, height=0.1, confidence=0.99)
    big_low_conf = RawFace(
        center_x=0.5, center_y=0.5, width=0.3, height=0.3, confidence=0.4
    )
    big_high_conf = RawFace(
        center_x=0.6, center_y=0.4, width=0.3, height=0.3, confidence=0.9
    )

    assert select_primary_face([small, big_low_conf, big_high_conf]) is big_high_conf


def test_normalize_motion_scores_scales_to_unit_interval() -> None:
    assert normalize_motion_scores([]) == []
    assert normalize_motion_scores([0.0, 0.0]) == [0.0, 0.0]
    normalized = normalize_motion_scores([0.0, 2.0, 4.0])
    assert normalized == [0.0, 0.5, 1.0]


def test_normalize_motion_scores_handles_nan_and_negative_values() -> None:
    normalized = normalize_motion_scores([math.nan, -1.0, 3.0])

    assert normalized[0] == 0.0
    assert normalized[1] == 0.0
    assert normalized[2] == 1.0


def test_smooth_face_trajectory_preserves_none_entries() -> None:
    samples = [_stub_sample(i * 0.5) for i in range(3)]
    faces = [_face(0.5, 0.5), None, _face(0.6, 0.6)]

    smoothed = smooth_face_trajectory(samples, faces, min_cutoff=1.0, beta=0.1)

    assert smoothed[0] is not None
    assert smoothed[1] is None
    assert smoothed[2] is not None


def test_smooth_face_trajectory_reduces_center_x_jitter() -> None:
    samples = [_stub_sample(i * 0.1) for i in range(10)]
    jittery_x = [0.5, 0.52, 0.48, 0.51, 0.49, 0.505, 0.495, 0.51, 0.49, 0.5]
    faces = [_face(x, 0.5) for x in jittery_x]

    smoothed = smooth_face_trajectory(samples, faces, min_cutoff=1.0, beta=0.0)

    raw_range = max(jittery_x) - min(jittery_x)
    smoothed_xs = [face.center_x for face in smoothed if face is not None]
    assert max(smoothed_xs) - min(smoothed_xs) < raw_range


def test_build_frames_round_trips_into_vision_timeline() -> None:
    samples = [_stub_sample(0.0), _stub_sample(0.5), _stub_sample(1.0)]
    faces = [_face(0.5, 0.4), _face(0.55, 0.42), None]
    motions = [0.0, 2.0, 1.0]
    shot_timestamps = [0.51]

    frames = build_frames(
        samples=samples,
        faces=faces,
        motion_magnitudes=motions,
        shot_timestamps=shot_timestamps,
    )
    timeline = build_vision_timeline({"frames": frames})

    assert timeline.frames[0].primary_face is not None
    assert timeline.frames[0].shot_change is False
    assert timeline.frames[1].shot_change is True
    assert timeline.frames[2].primary_face is None
    assert 0.0 <= timeline.frames[1].motion_score <= 1.0


def test_build_frames_rounds_motion_and_timestamps() -> None:
    samples = [_stub_sample(0.123456), _stub_sample(0.876543)]
    frames = build_frames(
        samples=samples,
        faces=[None, None],
        motion_magnitudes=[0.0, 1.2345678],
        shot_timestamps=[],
    )

    assert frames[0]["timestamp_seconds"] == round(0.123456, 3)
    assert frames[1]["motion_score"] == round(1.0, 4)


def test_extract_faces_reads_mediapipe_detection_shape() -> None:
    detection = SimpleNamespace(
        detections=[
            SimpleNamespace(
                location_data=SimpleNamespace(
                    relative_bounding_box=SimpleNamespace(
                        xmin=0.2, ymin=0.1, width=0.3, height=0.4
                    )
                ),
                score=[0.88],
            )
        ]
    )

    faces = _extract_faces(detection)

    assert len(faces) == 1
    face = faces[0]
    assert face.center_x == pytest.approx(0.35)
    assert face.center_y == pytest.approx(0.3)
    assert face.width == pytest.approx(0.3)
    assert face.height == pytest.approx(0.4)
    assert face.confidence == pytest.approx(0.88)


def test_extract_faces_skips_detections_without_bounding_box() -> None:
    detection = SimpleNamespace(
        detections=[SimpleNamespace(location_data=None, score=[0.9])]
    )

    assert _extract_faces(detection) == []


def test_analyze_composes_pipeline_stages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")
    samples = [_stub_sample(0.0), _stub_sample(0.5), _stub_sample(1.0)]
    faces = [_face(0.5, 0.5), _face(0.52, 0.48), None]
    motions = [0.0, 1.0, 2.0]
    shot_times = [0.5]

    calls: dict[str, object] = {}

    def fake_sample_frames(path, config):
        calls["sample.path"] = path
        calls["sample.fps"] = config.sample_fps
        return samples

    def fake_detect_shots(path, *, threshold):
        calls["shots.path"] = path
        calls["shots.threshold"] = threshold
        return shot_times

    def fake_detect_faces(samples_in, *, min_confidence):
        calls["faces.count"] = len(samples_in)
        calls["faces.min_confidence"] = min_confidence
        return faces

    def fake_motion(samples_in):
        calls["motion.count"] = len(samples_in)
        return motions

    monkeypatch.setattr("clipper.adapters.vision._sample_frames", fake_sample_frames)
    monkeypatch.setattr(
        "clipper.adapters.vision._detect_shot_changes", fake_detect_shots
    )
    monkeypatch.setattr(
        "clipper.adapters.vision._detect_faces_per_frame", fake_detect_faces
    )
    monkeypatch.setattr(
        "clipper.adapters.vision._compute_motion_per_frame", fake_motion
    )

    result = analyze(video, config=VisionConfig(sample_fps=4.0, scene_threshold=30.0))

    assert result["model"] == DEFAULT_MODEL
    assert len(result["frames"]) == 3
    assert result["frames"][0]["timestamp_seconds"] == 0.0
    assert result["frames"][1]["shot_change"] is True
    assert result["frames"][2]["primary_face"] is None
    assert calls["sample.fps"] == 4.0
    assert calls["shots.threshold"] == 30.0
    assert calls["faces.count"] == 3
    assert calls["motion.count"] == 3


def test_analyze_raises_when_no_samples_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")

    monkeypatch.setattr(
        "clipper.adapters.vision._sample_frames", lambda path, config: []
    )

    with pytest.raises(VisionError):
        analyze(video)


def test_analyze_module_exposes_default_model_constant() -> None:
    assert vision_adapter.DEFAULT_MODEL == "opencv-mediapipe-scenedetect"


def _stub_sample(timestamp: float) -> FrameSample:
    return FrameSample(timestamp_seconds=timestamp, rgb=object(), gray_small=object())


def _face(center_x: float, center_y: float) -> RawFace:
    return RawFace(
        center_x=center_x,
        center_y=center_y,
        width=0.3,
        height=0.4,
        confidence=0.9,
    )
