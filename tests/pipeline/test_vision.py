import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from clipper.pipeline.vision import build_vision_timeline


@pytest.fixture
def sample_face_payload() -> dict:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "sample_faces.json"
    return json.loads(fixture_path.read_text())


def test_build_vision_timeline_emits_face_and_motion_events(
    sample_face_payload: dict,
) -> None:
    timeline = build_vision_timeline(sample_face_payload)

    assert timeline.frames[0].primary_face.center_x == 0.48
    assert timeline.frames[0].motion_score == 0.72


def test_primary_face_is_available_for_crop_planning(
    sample_face_payload: dict,
) -> None:
    timeline = build_vision_timeline(sample_face_payload)

    assert timeline.frames[1].primary_face is not None
    assert 0.0 <= timeline.frames[1].primary_face.center_x <= 1.0


def test_build_vision_timeline_preserves_missing_primary_face(
    sample_face_payload: dict,
) -> None:
    timeline = build_vision_timeline(sample_face_payload)

    assert timeline.frames[2].primary_face is None


def test_build_vision_timeline_rejects_malformed_payload_shape() -> None:
    with pytest.raises(ValidationError):
        build_vision_timeline({"frames": [{"motion_score": 0.72}]})


def test_build_vision_timeline_rejects_stringly_typed_values(
    sample_face_payload: dict,
) -> None:
    sample_face_payload["frames"][0]["motion_score"] = "0.72"
    sample_face_payload["frames"][0]["shot_change"] = "false"

    with pytest.raises(ValidationError):
        build_vision_timeline(sample_face_payload)


def test_build_vision_timeline_rejects_out_of_range_normalized_values(
    sample_face_payload: dict,
) -> None:
    sample_face_payload["frames"][1]["primary_face"]["center_x"] = 1.2

    with pytest.raises(ValidationError):
        build_vision_timeline(sample_face_payload)
