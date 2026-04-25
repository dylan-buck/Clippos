import pytest

from clippos.pipeline.caption_styles import (
    ASS_ALIGN_BOTTOM_CENTER,
    ASS_ALIGN_MIDDLE_CENTER,
    ASS_ALIGN_TOP_CENTER,
    CAPTION_STYLES,
    resolve_caption_style,
)


def test_resolve_caption_style_returns_same_object_for_alias() -> None:
    assert resolve_caption_style("hook-default") is resolve_caption_style(
        "bottom-creator"
    )


def test_bottom_creator_matches_prior_defaults() -> None:
    style = resolve_caption_style("bottom-creator")

    # the original hardcoded renderer used 5% font size, 12% margin, bold, outline=3
    assert style.alignment == ASS_ALIGN_BOTTOM_CENTER
    assert style.bold is True
    assert style.font_size(1920) == 96
    assert style.margin_v(1920) == 230


def test_center_punch_aligns_center_with_large_font() -> None:
    style = resolve_caption_style("center-punch")

    assert style.alignment == ASS_ALIGN_MIDDLE_CENTER
    assert style.font_size(1920) > resolve_caption_style("bottom-creator").font_size(
        1920
    )
    # center punch sits at the middle of the frame — margin_v must not push it away
    assert style.margin_v(1920) == 0


def test_top_clean_aligns_top_and_clean() -> None:
    style = resolve_caption_style("top-clean")

    assert style.alignment == ASS_ALIGN_TOP_CENTER
    assert style.bold is False


def test_font_size_respects_floor_on_small_canvases() -> None:
    style = resolve_caption_style("bottom-creator")

    # a 400px-tall canvas would give 20px by ratio; floor clamps up to 32
    assert style.font_size(400) == style.font_size_floor


@pytest.mark.parametrize("preset", list(CAPTION_STYLES.keys()))
def test_every_preset_resolves_to_a_valid_style(preset) -> None:
    style = resolve_caption_style(preset)
    assert 1 <= style.alignment <= 9
    assert style.margin_l >= 0
    assert style.margin_r >= 0
    assert style.font_size_ratio > 0
