import pytest
from pydantic import ValidationError

from clipper.models.scoring import (
    ClipBrief,
    ClipScore,
    MiningSignals,
    RubricScores,
    ScoringRequest,
    ScoringResponse,
)


def _mining_signals(**overrides) -> MiningSignals:
    defaults = {
        "hook": 0.5,
        "keyword": 0.5,
        "numeric": 0.1,
        "interjection": 0.1,
        "payoff": 0.4,
        "question_to_answer": 0.0,
        "motion": 0.3,
        "shot_change": 0.2,
        "face_presence": 0.6,
        "speaker_interaction": 0.1,
        "delivery_variance": 0.2,
        "buried_lead": False,
        "dangling_question": False,
        "rambling_middle": False,
        "reasons": ["payoff"],
        "spike_categories": ["absurdity"],
    }
    defaults.update(overrides)
    return MiningSignals(**defaults)


def _clip_brief(**overrides) -> ClipBrief:
    defaults = {
        "clip_id": "clip-000",
        "clip_hash": "abc123def4567890",
        "start_seconds": 0.0,
        "end_seconds": 18.0,
        "duration_seconds": 18.0,
        "transcript": "Here's the thing. Turns out the outcome was wild.",
        "speakers": ["speaker_1"],
        "mining_score": 0.8,
        "mining_signals": _mining_signals(),
    }
    defaults.update(overrides)
    return ClipBrief(**defaults)


def _rubric_scores(**overrides) -> RubricScores:
    defaults = {
        "hook": 0.9,
        "shareability": 0.7,
        "standalone_clarity": 0.8,
        "payoff": 0.75,
        "delivery_energy": 0.65,
        "quotability": 0.55,
    }
    defaults.update(overrides)
    return RubricScores(**defaults)


def _clip_score(**overrides) -> ClipScore:
    defaults = {
        "clip_id": "clip-000",
        "clip_hash": "abc123def4567890",
        "rubric": _rubric_scores(),
        "spike_categories": ["taboo"],
        "penalties": [],
        "final_score": 0.82,
        "title": "Secret reveal",
        "hook": "Nobody tells you this",
        "reasons": ["strong hook", "payoff"],
    }
    defaults.update(overrides)
    return ClipScore(**defaults)


def test_clip_brief_rejects_end_at_or_before_start() -> None:
    with pytest.raises(ValidationError):
        _clip_brief(start_seconds=10.0, end_seconds=10.0, duration_seconds=0.1)
    with pytest.raises(ValidationError):
        _clip_brief(start_seconds=10.0, end_seconds=5.0, duration_seconds=0.1)


def test_clip_brief_rejects_non_positive_duration() -> None:
    with pytest.raises(ValidationError):
        _clip_brief(duration_seconds=0.0)


def test_mining_signals_rejects_values_outside_unit_interval() -> None:
    with pytest.raises(ValidationError):
        _mining_signals(hook=1.5)
    with pytest.raises(ValidationError):
        _mining_signals(payoff=-0.1)


def test_scoring_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ScoringRequest.model_validate(
            {
                "rubric_version": "1.0.0",
                "job_id": "abc",
                "video_path": "/tmp/input.mp4",
                "rubric_prompt": "prompt",
                "response_schema": {},
                "clips": [],
                "unknown": "field",
            }
        )


def test_rubric_scores_rejects_values_outside_unit_interval() -> None:
    with pytest.raises(ValidationError):
        _rubric_scores(hook=1.2)
    with pytest.raises(ValidationError):
        _rubric_scores(quotability=-0.01)


def test_clip_score_rejects_unknown_spike_category() -> None:
    with pytest.raises(ValidationError):
        _clip_score(spike_categories=["unknown_category"])


def test_clip_score_rejects_unknown_penalty() -> None:
    with pytest.raises(ValidationError):
        _clip_score(penalties=["not_a_penalty"])


def test_clip_score_rejects_final_score_outside_unit_interval() -> None:
    with pytest.raises(ValidationError):
        _clip_score(final_score=1.01)


def test_scoring_response_roundtrips_through_model_dump() -> None:
    response = ScoringResponse(
        rubric_version="1.0.0",
        job_id="abc",
        scores=[_clip_score()],
    )
    payload = response.model_dump(mode="json")
    restored = ScoringResponse.model_validate(payload)
    assert restored == response
