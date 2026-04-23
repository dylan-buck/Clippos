from __future__ import annotations

import math

import pytest

from clipper.adapters.one_euro import OneEuroFilter


def test_one_euro_returns_first_value_verbatim() -> None:
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)

    assert f(0.5, timestamp_seconds=0.0) == 0.5


def test_one_euro_reduces_jitter_in_stationary_signal() -> None:
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    noisy = [0.5, 0.52, 0.48, 0.51, 0.49, 0.505, 0.495]
    smoothed = [
        f(value, timestamp_seconds=index * 0.1) for index, value in enumerate(noisy)
    ]

    raw_range = max(noisy) - min(noisy)
    smoothed_range = max(smoothed) - min(smoothed)
    assert smoothed_range < raw_range


def test_one_euro_tracks_fast_motion_when_beta_is_high() -> None:
    slow = OneEuroFilter(min_cutoff=0.5, beta=0.0)
    responsive = OneEuroFilter(min_cutoff=0.5, beta=4.0)
    jumps = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    slow_out = [slow(v, timestamp_seconds=i * 0.1) for i, v in enumerate(jumps)]
    responsive_out = [
        responsive(v, timestamp_seconds=i * 0.1) for i, v in enumerate(jumps)
    ]

    assert responsive_out[-1] > slow_out[-1]


def test_one_euro_rejects_non_positive_cutoff() -> None:
    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff=0.0)
    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff=1.0, derivative_cutoff=-1.0)


def test_one_euro_handles_non_monotonic_timestamps() -> None:
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    f(0.4, timestamp_seconds=0.0)
    assert f(0.6, timestamp_seconds=-1.0) == 0.6


def test_one_euro_output_is_finite_for_finite_inputs() -> None:
    f = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    values = [0.1, 0.2, 0.4, 0.8, 0.6]
    for index, value in enumerate(values):
        assert math.isfinite(f(value, timestamp_seconds=index * 0.1))
