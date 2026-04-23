from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clipper.adapters.one_euro import OneEuroFilter

DEFAULT_MODEL = "opencv-mediapipe-scenedetect"


@dataclass(frozen=True)
class VisionConfig:
    sample_fps: float = 2.0
    scene_threshold: float = 27.0
    face_min_confidence: float = 0.5
    motion_frame_width: int = 256
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.1


class VisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrameSample:
    timestamp_seconds: float
    rgb: Any = field(repr=False)
    gray_small: Any = field(repr=False)


@dataclass(frozen=True)
class RawFace:
    center_x: float
    center_y: float
    width: float
    height: float
    confidence: float


def analyze(video_path: Path, *, config: VisionConfig | None = None) -> dict:
    cfg = config or VisionConfig()
    samples = _sample_frames(video_path, cfg)
    if not samples:
        raise VisionError(f"No frames could be sampled from {video_path}")
    shot_timestamps = _detect_shot_changes(video_path, threshold=cfg.scene_threshold)
    face_series = _detect_faces_per_frame(
        samples, min_confidence=cfg.face_min_confidence
    )
    motion_magnitudes = _compute_motion_per_frame(samples)
    smoothed_faces = smooth_face_trajectory(
        samples,
        face_series,
        min_cutoff=cfg.one_euro_min_cutoff,
        beta=cfg.one_euro_beta,
    )
    frames = build_frames(
        samples=samples,
        faces=smoothed_faces,
        motion_magnitudes=motion_magnitudes,
        shot_timestamps=shot_timestamps,
    )
    return {"model": DEFAULT_MODEL, "frames": frames}


def smooth_face_trajectory(
    samples: list[FrameSample],
    faces: list[RawFace | None],
    *,
    min_cutoff: float,
    beta: float,
) -> list[RawFace | None]:
    center_x_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
    center_y_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
    width_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
    height_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
    smoothed: list[RawFace | None] = []
    for sample, face in zip(samples, faces, strict=True):
        if face is None:
            smoothed.append(None)
            continue
        t = sample.timestamp_seconds
        smoothed.append(
            RawFace(
                center_x=_clamp01(center_x_filter(face.center_x, t)),
                center_y=_clamp01(center_y_filter(face.center_y, t)),
                width=_clamp01(width_filter(face.width, t)),
                height=_clamp01(height_filter(face.height, t)),
                confidence=face.confidence,
            )
        )
    return smoothed


def build_frames(
    *,
    samples: list[FrameSample],
    faces: list[RawFace | None],
    motion_magnitudes: list[float],
    shot_timestamps: list[float],
) -> list[dict]:
    normalized_motion = normalize_motion_scores(motion_magnitudes)
    shot_set = _shot_timestamp_lookup(samples, shot_timestamps)
    frames: list[dict] = []
    for sample, face, motion in zip(samples, faces, normalized_motion, strict=True):
        frames.append(
            {
                "timestamp_seconds": round(sample.timestamp_seconds, 3),
                "motion_score": round(motion, 4),
                "shot_change": sample.timestamp_seconds in shot_set,
                "primary_face": _serialize_face(face),
            }
        )
    return frames


def normalize_motion_scores(magnitudes: list[float]) -> list[float]:
    if not magnitudes:
        return []
    finite = [m for m in magnitudes if math.isfinite(m) and m >= 0]
    ceiling = max(finite) if finite else 0.0
    if ceiling <= 0:
        return [0.0 for _ in magnitudes]
    return [
        _clamp01(m / ceiling) if math.isfinite(m) and m >= 0 else 0.0
        for m in magnitudes
    ]


def select_primary_face(candidates: list[RawFace]) -> RawFace | None:
    if not candidates:
        return None
    return max(candidates, key=lambda f: f.width * f.height * f.confidence)


def _sample_frames(video_path: Path, config: VisionConfig) -> list[FrameSample]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VisionError(f"Unable to open video for vision analysis: {video_path}")
    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        if source_fps <= 0:
            raise VisionError(f"Video reports non-positive fps: {video_path}")
        stride = max(int(round(source_fps / config.sample_fps)), 1)
        samples: list[FrameSample] = []
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                timestamp_seconds = frame_index / source_fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray_small = _downscale_to_gray(frame, config.motion_frame_width)
                samples.append(
                    FrameSample(
                        timestamp_seconds=timestamp_seconds,
                        rgb=rgb,
                        gray_small=gray_small,
                    )
                )
            frame_index += 1
        return samples
    finally:
        capture.release()


def _downscale_to_gray(frame: Any, target_width: int) -> Any:
    import cv2

    height, width = frame.shape[:2]
    if width <= 0 or target_width <= 0:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale = target_width / width
    new_size = (target_width, max(int(round(height * scale)), 1))
    resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def _detect_shot_changes(video_path: Path, *, threshold: float) -> list[float]:
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))
    manager.detect_scenes(video=video)
    scenes = manager.get_scene_list()
    timestamps: list[float] = []
    for index, (start, _end) in enumerate(scenes):
        if index == 0:
            continue
        timestamps.append(float(start.get_seconds()))
    return timestamps


def _detect_faces_per_frame(
    samples: list[FrameSample], *, min_confidence: float
) -> list[RawFace | None]:
    import mediapipe as mp

    results: list[RawFace | None] = []
    detector_ctx = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=min_confidence
    )
    with detector_ctx as detector:
        for sample in samples:
            detection = detector.process(sample.rgb)
            faces = _extract_faces(detection)
            results.append(select_primary_face(faces))
    return results


def _extract_faces(detection: Any) -> list[RawFace]:
    raw_detections = getattr(detection, "detections", None) or []
    faces: list[RawFace] = []
    for item in raw_detections:
        location = getattr(item, "location_data", None)
        bbox = getattr(location, "relative_bounding_box", None) if location else None
        if bbox is None:
            continue
        xmin = float(getattr(bbox, "xmin", 0.0) or 0.0)
        ymin = float(getattr(bbox, "ymin", 0.0) or 0.0)
        width = float(getattr(bbox, "width", 0.0) or 0.0)
        height = float(getattr(bbox, "height", 0.0) or 0.0)
        score_values = list(getattr(item, "score", []) or [])
        confidence = float(score_values[0]) if score_values else 0.0
        faces.append(
            RawFace(
                center_x=_clamp01(xmin + width / 2.0),
                center_y=_clamp01(ymin + height / 2.0),
                width=_clamp01(width),
                height=_clamp01(height),
                confidence=_clamp01(confidence),
            )
        )
    return faces


def _compute_motion_per_frame(samples: list[FrameSample]) -> list[float]:
    import cv2
    import numpy as np

    magnitudes: list[float] = [0.0]
    for previous, current in zip(samples, samples[1:], strict=False):
        flow = cv2.calcOpticalFlowFarneback(
            previous.gray_small,
            current.gray_small,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        magnitude = float(np.linalg.norm(flow, axis=2).mean())
        magnitudes.append(magnitude)
    return magnitudes


def _shot_timestamp_lookup(
    samples: list[FrameSample], shot_timestamps: list[float]
) -> set[float]:
    if not samples or not shot_timestamps:
        return set()
    sample_times = [sample.timestamp_seconds for sample in samples]
    hits: set[float] = set()
    for shot in shot_timestamps:
        nearest = min(sample_times, key=lambda t: abs(t - shot))
        hits.add(nearest)
    return hits


def _serialize_face(face: RawFace | None) -> dict | None:
    if face is None:
        return None
    return {
        "center_x": round(face.center_x, 4),
        "center_y": round(face.center_y, 4),
        "width": round(face.width, 4),
        "height": round(face.height, 4),
        "confidence": round(face.confidence, 4),
    }


def _clamp01(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return min(max(value, 0.0), 1.0)
