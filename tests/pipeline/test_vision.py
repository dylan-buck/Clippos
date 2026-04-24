import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from clipper.adapters.vision import VisionConfig
from clipper.pipeline.vision import (
    VISION_CACHE_FILENAME,
    build_vision_timeline,
    run_vision,
)


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


def _fake_analyzer_result(sample_face_payload: dict) -> dict:
    return {
        "model": "retinaface-resnet50-raft-scenedetect",
        "frames": sample_face_payload["frames"],
    }


def test_run_vision_writes_cache_and_returns_payload(
    tmp_path: Path,
    sample_face_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls: list[Path] = []

    def fake_analyze(path: Path, *, config: VisionConfig) -> dict:
        calls.append(path)
        return _fake_analyzer_result(sample_face_payload)

    monkeypatch.setattr("clipper.pipeline.vision.vision_adapter.analyze", fake_analyze)

    payload = run_vision(video, workspace)

    assert calls == [video]
    assert payload == {"frames": sample_face_payload["frames"]}
    cache_path = workspace / VISION_CACHE_FILENAME
    assert cache_path.exists()
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cached["metadata"] == {"model": "retinaface-resnet50-raft-scenedetect"}
    assert cached["payload"] == payload


def test_run_vision_round_trips_into_timeline(
    tmp_path: Path,
    sample_face_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setattr(
        "clipper.pipeline.vision.vision_adapter.analyze",
        lambda path, *, config: _fake_analyzer_result(sample_face_payload),
    )

    payload = run_vision(video, workspace)
    timeline = build_vision_timeline(payload)

    assert timeline.frames[0].motion_score == 0.72
    assert timeline.frames[2].primary_face is None


def test_run_vision_reuses_cache_when_model_matches(
    tmp_path: Path,
    sample_face_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache_path = workspace / VISION_CACHE_FILENAME
    cache_path.write_text(
        json.dumps(
            {
                "metadata": {"model": "retinaface-resnet50-raft-scenedetect"},
                "payload": {"frames": sample_face_payload["frames"]},
            }
        ),
        encoding="utf-8",
    )

    def explode(path: Path, *, config: VisionConfig) -> dict:
        raise AssertionError("adapter should not be called on cache hit")

    monkeypatch.setattr("clipper.pipeline.vision.vision_adapter.analyze", explode)

    payload = run_vision(video, workspace)

    assert payload == {"frames": sample_face_payload["frames"]}


def test_run_vision_reruns_when_cached_model_mismatches(
    tmp_path: Path,
    sample_face_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache_path = workspace / VISION_CACHE_FILENAME
    cache_path.write_text(
        json.dumps(
            {
                "metadata": {"model": "legacy"},
                "payload": {"frames": sample_face_payload["frames"]},
            }
        ),
        encoding="utf-8",
    )
    calls: list[Path] = []

    def fake_analyze(path: Path, *, config: VisionConfig) -> dict:
        calls.append(path)
        return _fake_analyzer_result(sample_face_payload)

    monkeypatch.setattr("clipper.pipeline.vision.vision_adapter.analyze", fake_analyze)

    payload = run_vision(video, workspace)

    assert calls == [video]
    assert payload == {"frames": sample_face_payload["frames"]}
    refreshed = json.loads(cache_path.read_text(encoding="utf-8"))
    assert refreshed["metadata"]["model"] == "retinaface-resnet50-raft-scenedetect"


def test_run_vision_reruns_when_cache_is_corrupt(
    tmp_path: Path,
    sample_face_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache_path = workspace / VISION_CACHE_FILENAME
    cache_path.write_text("not json", encoding="utf-8")

    monkeypatch.setattr(
        "clipper.pipeline.vision.vision_adapter.analyze",
        lambda path, *, config: _fake_analyzer_result(sample_face_payload),
    )

    payload = run_vision(video, workspace)

    assert payload == {"frames": sample_face_payload["frames"]}
    refreshed = json.loads(cache_path.read_text(encoding="utf-8"))
    assert refreshed["metadata"]["model"] == "retinaface-resnet50-raft-scenedetect"


def test_run_vision_reruns_when_cache_shape_is_wrong(
    tmp_path: Path,
    sample_face_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cache_path = workspace / VISION_CACHE_FILENAME
    cache_path.write_text(
        json.dumps({"frames": sample_face_payload["frames"]}), encoding="utf-8"
    )

    monkeypatch.setattr(
        "clipper.pipeline.vision.vision_adapter.analyze",
        lambda path, *, config: _fake_analyzer_result(sample_face_payload),
    )

    payload = run_vision(video, workspace)

    assert payload == {"frames": sample_face_payload["frames"]}
    refreshed = json.loads(cache_path.read_text(encoding="utf-8"))
    assert "metadata" in refreshed and "payload" in refreshed
