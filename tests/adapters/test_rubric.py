from clipper.adapters.rubric import (
    RUBRIC_PROMPT,
    RUBRIC_VERSION,
    build_response_schema,
    build_rubric_prompt,
)


def test_rubric_prompt_covers_all_rubric_dimensions() -> None:
    prompt = build_rubric_prompt()
    for dimension in (
        "hook",
        "shareability",
        "standalone_clarity",
        "payoff",
        "delivery_energy",
        "quotability",
    ):
        assert dimension in prompt, f"missing rubric dimension: {dimension}"


def test_rubric_prompt_covers_all_spike_categories() -> None:
    prompt = build_rubric_prompt()
    for category in (
        "emotional_confrontation",
        "controversy",
        "taboo",
        "absurdity",
        "action",
        "unusually_useful_claim",
        # M4 (docs/miner-quality.md) — interview / podcast vertical.
        "expert_endorsement",
        "specific_pick",
        "big_number",
    ):
        assert category in prompt, f"missing spike category: {category}"


def test_rubric_prompt_covers_all_penalties() -> None:
    prompt = build_rubric_prompt()
    for penalty in (
        "buried_lead",
        "dangling_question",
        "rambling_middle",
        "context_dependent",
        "low_delivery",
    ):
        assert penalty in prompt, f"missing penalty: {penalty}"


def test_rubric_prompt_is_stable_across_calls() -> None:
    assert build_rubric_prompt() == RUBRIC_PROMPT
    assert build_rubric_prompt() is build_rubric_prompt()


def test_build_response_schema_pins_rubric_version_const() -> None:
    schema = build_response_schema()
    assert schema["properties"]["rubric_version"]["const"] == RUBRIC_VERSION


def test_build_response_schema_locks_spike_and_penalty_enums() -> None:
    schema = build_response_schema()
    score_item = schema["properties"]["scores"]["items"]
    assert set(score_item["properties"]["spike_categories"]["items"]["enum"]) == {
        "emotional_confrontation",
        "controversy",
        "taboo",
        "absurdity",
        "action",
        "unusually_useful_claim",
        # M4 (docs/miner-quality.md) — interview / podcast vertical.
        "expert_endorsement",
        "specific_pick",
        "big_number",
    }
    assert set(score_item["properties"]["penalties"]["items"]["enum"]) == {
        "buried_lead",
        "dangling_question",
        "rambling_middle",
        "context_dependent",
        "low_delivery",
    }


def test_build_response_schema_requires_all_per_clip_fields() -> None:
    schema = build_response_schema()
    score_item = schema["properties"]["scores"]["items"]
    assert set(score_item["required"]) == {
        "clip_id",
        "clip_hash",
        "rubric",
        "spike_categories",
        "penalties",
        "final_score",
        "title",
        "hook",
        "reasons",
    }
