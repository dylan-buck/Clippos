"""Named caption presets for the ASS subtitle renderer.

Inspired by mkreel (github.com/adidshaft/mkreel): instead of a single hardcoded
style, the render stage picks from a small catalog of position/size/style
combinations. Each preset collapses ASS knobs — alignment, margins, font-size
ratio, outline, shadow, bold — into one ``CaptionStyle`` resolved at render
time.

``hook-default`` is kept as an alias of ``bottom-creator`` so existing
``ClipposJob`` fixtures and cached manifests keep working after the widen.
"""

from __future__ import annotations

from dataclasses import dataclass

from clippos.models.render import CaptionPreset

ASS_ALIGN_BOTTOM_CENTER = 2
ASS_ALIGN_MIDDLE_CENTER = 5
ASS_ALIGN_TOP_CENTER = 8


@dataclass(frozen=True)
class CaptionStyle:
    """Resolved ASS style knobs for a single caption preset.

    All size/margin ratios are fractions of the canonical output height (1920 px
    for 9:16, 1080 px for 1:1 / 16:9). Floors keep captions legible on the
    smaller 1:1 / 16:9 canvases.
    """

    alignment: int
    bold: bool
    font_size_ratio: float
    font_size_floor: int
    margin_v_ratio: float
    margin_v_floor: int
    margin_l: int
    margin_r: int
    outline: float
    shadow: float

    def font_size(self, play_height: int) -> int:
        return max(int(round(play_height * self.font_size_ratio)), self.font_size_floor)

    def margin_v(self, play_height: int) -> int:
        return max(int(round(play_height * self.margin_v_ratio)), self.margin_v_floor)


CAPTION_STYLES: dict[CaptionPreset, CaptionStyle] = {
    "bottom-creator": CaptionStyle(
        alignment=ASS_ALIGN_BOTTOM_CENTER,
        bold=True,
        font_size_ratio=0.05,
        font_size_floor=32,
        margin_v_ratio=0.12,
        margin_v_floor=80,
        margin_l=60,
        margin_r=60,
        outline=3,
        shadow=2,
    ),
    "bottom-compact": CaptionStyle(
        alignment=ASS_ALIGN_BOTTOM_CENTER,
        bold=True,
        font_size_ratio=0.04,
        font_size_floor=28,
        margin_v_ratio=0.10,
        margin_v_floor=64,
        margin_l=72,
        margin_r=72,
        outline=2.5,
        shadow=1.5,
    ),
    "lower-third-clean": CaptionStyle(
        alignment=ASS_ALIGN_BOTTOM_CENTER,
        bold=False,
        font_size_ratio=0.045,
        font_size_floor=30,
        margin_v_ratio=0.22,
        margin_v_floor=140,
        margin_l=80,
        margin_r=80,
        outline=2,
        shadow=1,
    ),
    "center-punch": CaptionStyle(
        alignment=ASS_ALIGN_MIDDLE_CENTER,
        bold=True,
        font_size_ratio=0.075,
        font_size_floor=48,
        margin_v_ratio=0.0,
        margin_v_floor=0,
        margin_l=48,
        margin_r=48,
        outline=4,
        shadow=3,
    ),
    "top-clean": CaptionStyle(
        alignment=ASS_ALIGN_TOP_CENTER,
        bold=False,
        font_size_ratio=0.045,
        font_size_floor=30,
        margin_v_ratio=0.10,
        margin_v_floor=64,
        margin_l=80,
        margin_r=80,
        outline=2,
        shadow=1,
    ),
}

_ALIASES: dict[CaptionPreset, CaptionPreset] = {
    "hook-default": "bottom-creator",
}


def resolve_caption_style(preset: CaptionPreset) -> CaptionStyle:
    canonical = _ALIASES.get(preset, preset)
    try:
        return CAPTION_STYLES[canonical]
    except KeyError as exc:  # pragma: no cover - Literal narrows this out
        raise ValueError(f"unknown caption preset: {preset!r}") from exc
