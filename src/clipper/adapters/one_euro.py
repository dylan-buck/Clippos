from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class _LowPass:
    alpha: float
    _previous: float | None = None

    def filter(self, value: float) -> float:
        if self._previous is None:
            self._previous = value
            return value
        filtered = self.alpha * value + (1.0 - self.alpha) * self._previous
        self._previous = filtered
        return filtered


class OneEuroFilter:
    """OneEuro filter (Casiez et al., 2012) for smoothing noisy time series.

    Reduces jitter at low speeds while tracking fast motion. See
    https://cristal.univ-lille.fr/~casiez/1euro/.
    """

    def __init__(
        self,
        *,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        derivative_cutoff: float = 1.0,
    ) -> None:
        if min_cutoff <= 0 or derivative_cutoff <= 0:
            raise ValueError("cutoff frequencies must be positive")
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._derivative_cutoff = derivative_cutoff
        self._value_filter: _LowPass | None = None
        self._derivative_filter = _LowPass(alpha=0.0)
        self._last_time: float | None = None

    def __call__(self, value: float, timestamp_seconds: float) -> float:
        if self._last_time is None or timestamp_seconds <= self._last_time:
            self._last_time = timestamp_seconds
            self._value_filter = _LowPass(alpha=_alpha(self._min_cutoff, dt=1.0))
            self._value_filter.filter(value)
            return value

        dt = timestamp_seconds - self._last_time
        self._last_time = timestamp_seconds

        derivative = (value - self._value_filter._previous) / dt  # type: ignore[union-attr]
        self._derivative_filter.alpha = _alpha(self._derivative_cutoff, dt=dt)
        derivative_filtered = self._derivative_filter.filter(derivative)

        cutoff = self._min_cutoff + self._beta * abs(derivative_filtered)
        self._value_filter.alpha = _alpha(cutoff, dt=dt)  # type: ignore[union-attr]
        return self._value_filter.filter(value)  # type: ignore[union-attr]


def _alpha(cutoff: float, *, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)
