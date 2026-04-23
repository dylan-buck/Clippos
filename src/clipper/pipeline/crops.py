from __future__ import annotations

from fractions import Fraction

from clipper.adapters.one_euro import OneEuroFilter
from clipper.models.media import AspectRatio
from clipper.models.render import CropAnchor, CropPlan
from clipper.pipeline.vision import VisionTimeline

DEFAULT_RATIOS: tuple[AspectRatio, ...] = ("9:16", "1:1", "16:9")

_MIN_CUTOFF = 0.6
_BETA = 0.05
_DERIVATIVE_CUTOFF = 1.0

_RATIO_FRACTIONS: dict[AspectRatio, Fraction] = {
    "9:16": Fraction(9, 16),
    "1:1": Fraction(1, 1),
    "16:9": Fraction(16, 9),
}


def build_crop_plans(
    vision: VisionTimeline,
    *,
    start_seconds: float,
    end_seconds: float,
    source_width: int,
    source_height: int,
    ratios: tuple[AspectRatio, ...] = DEFAULT_RATIOS,
) -> dict[AspectRatio, CropPlan]:
    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds")
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source dimensions must be positive")
    if not ratios:
        raise ValueError("ratios must not be empty")

    anchors = _build_smoothed_anchors(
        vision, start_seconds=start_seconds, end_seconds=end_seconds
    )
    return {
        ratio: CropPlan(
            aspect_ratio=ratio,
            source_width=source_width,
            source_height=source_height,
            target_width=_target_width(ratio, source_width, source_height),
            target_height=_target_height(ratio, source_width, source_height),
            anchors=list(anchors),
        )
        for ratio in ratios
    }


def _build_smoothed_anchors(
    vision: VisionTimeline,
    *,
    start_seconds: float,
    end_seconds: float,
) -> list[CropAnchor]:
    raw_anchors = _collect_raw_anchors(
        vision, start_seconds=start_seconds, end_seconds=end_seconds
    )
    if not raw_anchors:
        return [
            CropAnchor(timestamp_seconds=0.0, center_x=0.5, center_y=0.5),
            CropAnchor(
                timestamp_seconds=end_seconds - start_seconds,
                center_x=0.5,
                center_y=0.5,
            ),
        ]

    x_filter = OneEuroFilter(
        min_cutoff=_MIN_CUTOFF,
        beta=_BETA,
        derivative_cutoff=_DERIVATIVE_CUTOFF,
    )
    y_filter = OneEuroFilter(
        min_cutoff=_MIN_CUTOFF,
        beta=_BETA,
        derivative_cutoff=_DERIVATIVE_CUTOFF,
    )

    smoothed: list[CropAnchor] = []
    for timestamp, center_x, center_y in raw_anchors:
        smoothed_x = x_filter(center_x, timestamp)
        smoothed_y = y_filter(center_y, timestamp)
        smoothed.append(
            CropAnchor(
                timestamp_seconds=timestamp,
                center_x=_clamp_unit(smoothed_x),
                center_y=_clamp_unit(smoothed_y),
            )
        )
    return smoothed


def _collect_raw_anchors(
    vision: VisionTimeline,
    *,
    start_seconds: float,
    end_seconds: float,
) -> list[tuple[float, float, float]]:
    raw: list[tuple[float, float, float]] = []
    for frame in vision.frames:
        if (
            frame.timestamp_seconds < start_seconds
            or frame.timestamp_seconds > end_seconds
        ):
            continue
        relative = frame.timestamp_seconds - start_seconds
        face = frame.primary_face
        if face is None:
            raw.append((relative, 0.5, 0.5))
        else:
            raw.append((relative, face.center_x, face.center_y))
    raw.sort(key=lambda item: item[0])
    return raw


def _target_width(ratio: AspectRatio, source_width: int, source_height: int) -> int:
    target_ratio = _RATIO_FRACTIONS[ratio]
    source_ratio = Fraction(source_width, source_height)
    if target_ratio >= source_ratio:
        return _even(source_width)
    return _even(int(source_height * float(target_ratio)))


def _target_height(ratio: AspectRatio, source_width: int, source_height: int) -> int:
    target_ratio = _RATIO_FRACTIONS[ratio]
    source_ratio = Fraction(source_width, source_height)
    if target_ratio >= source_ratio:
        return _even(int(source_width / float(target_ratio)))
    return _even(source_height)


def _even(value: int) -> int:
    if value <= 0:
        return 2
    return value if value % 2 == 0 else value - 1


def _clamp_unit(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
