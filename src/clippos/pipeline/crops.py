"""Crop plan generation with safe-zone subject tracking.

The smoothing model is a "heavy tripod" ported from openshorts
(github.com/mutonby/openshorts): the virtual camera holds still while the
subject stays inside a safe zone around the current center, then pans at a
controlled rate when the subject moves out of it. A shot-change forces an
immediate snap so we don't drift through a cut.

Rates are expressed in normalized-units-per-second instead of openshorts'
pixels-per-frame so the behavior is independent of source resolution and
vision-adapter sample rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from clippos.models.media import AspectRatio
from clippos.models.render import CropAnchor, CropPlan
from clippos.pipeline.vision import VisionTimeline

DEFAULT_RATIOS: tuple[AspectRatio, ...] = ("9:16", "1:1", "16:9")

SAFE_ZONE_RADIUS = 0.08
SLOW_PAN_RATE = 0.05
FAST_PAN_RATE = 0.25
FAST_PAN_THRESHOLD = 0.15
ANCHOR_DEDUPE_EPSILON = 0.005

_RATIO_FRACTIONS: dict[AspectRatio, Fraction] = {
    "9:16": Fraction(9, 16),
    "1:1": Fraction(1, 1),
    "16:9": Fraction(16, 9),
}


@dataclass
class SmoothedCameraman:
    """Heavy-tripod tracker operating in normalized [0, 1] coordinates.

    Call :meth:`step` once per vision frame in chronological order. The
    tracker holds its current center while ``target`` stays inside the safe
    zone, then pans at :attr:`slow_pan_rate` (or :attr:`fast_pan_rate` for
    large deltas) until it reaches the target. ``force_snap`` teleports the
    center to the target on shot changes and on the first frame.
    """

    safe_zone_radius: float = SAFE_ZONE_RADIUS
    slow_pan_rate: float = SLOW_PAN_RATE
    fast_pan_rate: float = FAST_PAN_RATE
    fast_pan_threshold: float = FAST_PAN_THRESHOLD
    current_x: float = 0.5
    current_y: float = 0.5
    target_x: float = 0.5
    target_y: float = 0.5
    _last_timestamp: float | None = None

    def update_target(self, target_x: float, target_y: float) -> None:
        self.target_x = _clamp_unit(target_x)
        self.target_y = _clamp_unit(target_y)

    def step(
        self, timestamp: float, *, force_snap: bool = False
    ) -> tuple[float, float]:
        if force_snap or self._last_timestamp is None:
            self.current_x = self.target_x
            self.current_y = self.target_y
            self._last_timestamp = timestamp
            return self.current_x, self.current_y

        dt = max(timestamp - self._last_timestamp, 0.0)
        self.current_x = self._advance(self.current_x, self.target_x, dt)
        self.current_y = self._advance(self.current_y, self.target_y, dt)
        self._last_timestamp = timestamp
        return self.current_x, self.current_y

    def _advance(self, current: float, target: float, dt: float) -> float:
        diff = target - current
        if abs(diff) <= self.safe_zone_radius:
            return current
        rate = (
            self.fast_pan_rate
            if abs(diff) > self.fast_pan_threshold
            else self.slow_pan_rate
        )
        step = rate * dt
        if step >= abs(diff):
            return target
        return current + (step if diff > 0 else -step)


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
    frames = [
        frame
        for frame in vision.frames
        if start_seconds <= frame.timestamp_seconds <= end_seconds
    ]
    frames.sort(key=lambda frame: frame.timestamp_seconds)

    has_any_face = any(frame.primary_face is not None for frame in frames)
    if not frames or not has_any_face:
        return [
            CropAnchor(timestamp_seconds=0.0, center_x=0.5, center_y=0.5),
            CropAnchor(
                timestamp_seconds=end_seconds - start_seconds,
                center_x=0.5,
                center_y=0.5,
            ),
        ]

    cameraman = SmoothedCameraman()
    first_face = next(frame for frame in frames if frame.primary_face is not None)
    cameraman.update_target(
        first_face.primary_face.center_x,  # type: ignore[union-attr]
        first_face.primary_face.center_y,  # type: ignore[union-attr]
    )

    tracked: list[CropAnchor] = []
    for frame in frames:
        relative = frame.timestamp_seconds - start_seconds
        face = frame.primary_face
        if face is not None:
            cameraman.update_target(face.center_x, face.center_y)
        center_x, center_y = cameraman.step(
            relative, force_snap=frame.shot_change or not tracked
        )
        tracked.append(
            CropAnchor(
                timestamp_seconds=relative,
                center_x=_clamp_unit(center_x),
                center_y=_clamp_unit(center_y),
            )
        )

    return _dedupe_runs(tracked)


def _dedupe_runs(anchors: list[CropAnchor]) -> list[CropAnchor]:
    """Collapse consecutive anchors within ``ANCHOR_DEDUPE_EPSILON`` into the
    first of the run. Ffmpeg expressions scale with anchor count, so trimming
    held-still runs keeps the filter graph tight."""
    if len(anchors) <= 2:
        return list(anchors)
    kept: list[CropAnchor] = [anchors[0]]
    for anchor in anchors[1:-1]:
        last = kept[-1]
        if (
            abs(anchor.center_x - last.center_x) < ANCHOR_DEDUPE_EPSILON
            and abs(anchor.center_y - last.center_y) < ANCHOR_DEDUPE_EPSILON
        ):
            continue
        kept.append(anchor)
    kept.append(anchors[-1])
    return kept


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
