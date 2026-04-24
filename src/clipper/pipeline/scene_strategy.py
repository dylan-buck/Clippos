"""Per-clip scene strategy selection.

Inspired by the openshorts TRACK-vs-GENERAL split (github.com/mutonby/openshorts):
scenes that contain a clear single subject are cropped with subject tracking;
scenes with no face or many faces fall back to a blurred-background, fit-width
composition that looks intentional instead of guessing a crop origin.

v1 selects a *single* mode per clip by pooling every vision frame in the
clip window. Per-scene-within-clip strategy (which would require ffmpeg
concat plumbing at render time) is deliberately deferred.
"""

from __future__ import annotations

from clipper.models.render import SceneMode
from clipper.pipeline.vision import VisionTimeline

TRACK_PRESENCE_THRESHOLD = 0.5


def derive_clip_mode(
    vision: VisionTimeline,
    *,
    start_seconds: float,
    end_seconds: float,
    presence_threshold: float = TRACK_PRESENCE_THRESHOLD,
) -> SceneMode:
    """Choose TRACK or GENERAL for a clip window.

    TRACK when a primary face is detected in at least ``presence_threshold``
    fraction of in-window frames. GENERAL when faces are mostly absent
    (landscape, group shots, B-roll) so the renderer can switch to the
    blurred-background composition.
    """
    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds")
    if not 0.0 <= presence_threshold <= 1.0:
        raise ValueError("presence_threshold must be between 0 and 1")

    frames_in_window = [
        frame
        for frame in vision.frames
        if start_seconds <= frame.timestamp_seconds <= end_seconds
    ]
    if not frames_in_window:
        return "GENERAL"

    with_face = sum(1 for frame in frames_in_window if frame.primary_face is not None)
    presence_ratio = with_face / len(frames_in_window)
    return "TRACK" if presence_ratio >= presence_threshold else "GENERAL"
