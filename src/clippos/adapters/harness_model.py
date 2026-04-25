from __future__ import annotations

from typing import Protocol


class HarnessModelAdapter(Protocol):
    def score_candidates(self, prompts: list[dict]) -> list[dict]: ...
