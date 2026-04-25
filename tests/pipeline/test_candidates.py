from __future__ import annotations

import pytest

from clippos.pipeline.candidates import (
    DurationPolicy,
    MiningConfig,
    ScoringWeights,
    WindowSignals,
    _detect_interview_blocks,
    derive_reasons,
    derive_spike_categories,
    enumerate_segment_windows,
    generate_candidates,
    has_buried_lead,
    has_dangling_question,
    is_rambling_middle,
    score_delivery_variance,
    score_face_presence,
    score_hook_strength,
    score_interjection_density,
    score_interview_keyword_spike,
    score_keyword_spike,
    score_motion_density,
    score_numeric_density,
    score_payoff_signal,
    score_question_to_answer,
    score_shot_change_density,
    score_speaker_interaction,
)
from clippos.pipeline.transcribe import build_transcript_timeline
from clippos.pipeline.vision import build_vision_timeline


@pytest.fixture
def long_transcript_timeline():
    return build_transcript_timeline(
        {
            "segments": [
                {
                    "speaker": "speaker_1",
                    "start_seconds": 0.0,
                    "end_seconds": 6.0,
                    "text": "Here's the thing: this crazy secret could ban your favorite snack forever.",
                    "words": [],
                },
                {
                    "speaker": "speaker_1",
                    "start_seconds": 6.0,
                    "end_seconds": 12.0,
                    "text": "The setup feels calm before the payoff lands.",
                    "words": [],
                },
                {
                    "speaker": "speaker_2",
                    "start_seconds": 12.0,
                    "end_seconds": 18.0,
                    "text": "Then the fight went wild and the room changed instantly.",
                    "words": [],
                },
                {
                    "speaker": "speaker_2",
                    "start_seconds": 18.0,
                    "end_seconds": 24.0,
                    "text": "This only makes sense if you watched the previous ten minutes.",
                    "words": [],
                },
                {
                    "speaker": "speaker_1",
                    "start_seconds": 24.0,
                    "end_seconds": 30.0,
                    "text": "Turns out the final explanation has a clear outcome.",
                    "words": [],
                },
            ]
        }
    )


@pytest.fixture
def long_vision_timeline():
    return build_vision_timeline(
        {
            "frames": [
                {"timestamp_seconds": 1.0, "motion_score": 0.82, "shot_change": True},
                {"timestamp_seconds": 4.0, "motion_score": 0.74, "shot_change": False},
                {"timestamp_seconds": 7.0, "motion_score": 0.35, "shot_change": False},
                {"timestamp_seconds": 10.0, "motion_score": 0.41, "shot_change": True},
                {"timestamp_seconds": 13.0, "motion_score": 0.78, "shot_change": True},
                {"timestamp_seconds": 16.0, "motion_score": 0.72, "shot_change": False},
                {"timestamp_seconds": 19.0, "motion_score": 0.18, "shot_change": False},
                {"timestamp_seconds": 22.0, "motion_score": 0.12, "shot_change": False},
                {"timestamp_seconds": 25.0, "motion_score": 0.32, "shot_change": False},
                {"timestamp_seconds": 28.0, "motion_score": 0.27, "shot_change": False},
            ]
        }
    )


def test_duration_policy_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        DurationPolicy(min_seconds=0.0)
    with pytest.raises(ValueError):
        DurationPolicy(min_seconds=30.0, max_seconds=15.0)


def test_enumerate_segment_windows_respects_duration_policy(
    long_transcript_timeline,
) -> None:
    windows = enumerate_segment_windows(
        long_transcript_timeline.segments,
        DurationPolicy(min_seconds=15.0, max_seconds=45.0),
    )

    durations = [w[-1].end_seconds - w[0].start_seconds for w in windows]
    assert all(15.0 <= d <= 45.0 for d in durations)


def test_enumerate_segment_windows_includes_full_coverage(
    long_transcript_timeline,
) -> None:
    windows = enumerate_segment_windows(
        long_transcript_timeline.segments,
        DurationPolicy(min_seconds=15.0, max_seconds=45.0),
    )

    starts = {w[0].start_seconds for w in windows}
    assert starts == {0.0, 6.0, 12.0}


def test_score_hook_strength_rewards_opening_phrase() -> None:
    strong = score_hook_strength("Here's the thing: this crazy secret blew up.")
    weak = score_hook_strength("The report was submitted on time.")

    assert strong > weak
    assert strong > 0.3


def test_score_keyword_spike_sums_categories() -> None:
    controversy_and_taboo = score_keyword_spike("the ban on this secret was wild")
    none = score_keyword_spike("we then continued the meeting")

    assert controversy_and_taboo > 0
    assert none == 0


def test_score_numeric_density_counts_digits() -> None:
    assert score_numeric_density("he ran 5 miles in 30 minutes") > 0
    assert score_numeric_density("the room was empty") == 0
    assert score_numeric_density("") == 0


def test_score_interjection_density_detects_known_tokens() -> None:
    assert score_interjection_density("wait, listen, honestly") > 0
    assert score_interjection_density("the sky is blue") == 0


def test_score_payoff_signal_matches_phrases_and_keywords() -> None:
    assert score_payoff_signal("turns out the result was stunning") > 0
    assert score_payoff_signal("and the outcome was amazing") > 0
    assert score_payoff_signal("no signal here") == 0


def test_score_question_to_answer_requires_question_in_first_half() -> None:
    from clippos.pipeline.transcribe import build_transcript_timeline

    timeline = build_transcript_timeline(
        {
            "segments": [
                {
                    "speaker": "a",
                    "start_seconds": 0.0,
                    "end_seconds": 2.0,
                    "text": "Why did the experiment fail?",
                    "words": [],
                },
                {
                    "speaker": "b",
                    "start_seconds": 2.0,
                    "end_seconds": 4.0,
                    "text": "Because the reagent was expired.",
                    "words": [],
                },
            ]
        }
    )

    assert score_question_to_answer(tuple(timeline.segments)) == 1.0


def test_score_motion_density_averages_in_window(long_vision_timeline) -> None:
    motion = score_motion_density(long_vision_timeline.frames, 0.0, 12.0)
    empty = score_motion_density(long_vision_timeline.frames, 500.0, 600.0)

    assert 0.0 < motion <= 1.0
    assert empty == 0.0


def test_score_shot_change_density_detects_cuts(long_vision_timeline) -> None:
    assert score_shot_change_density(long_vision_timeline.frames, 0.0, 15.0) > 0
    assert score_shot_change_density(long_vision_timeline.frames, 18.0, 30.0) == 0


def test_score_face_presence_returns_zero_when_no_frames(long_vision_timeline) -> None:
    assert score_face_presence(long_vision_timeline.frames, 500.0, 600.0) == 0


def test_score_speaker_interaction_rewards_back_and_forth(
    long_transcript_timeline,
) -> None:
    single_speaker = score_speaker_interaction(
        tuple(long_transcript_timeline.segments[:2])
    )
    mixed = score_speaker_interaction(tuple(long_transcript_timeline.segments))

    assert single_speaker == 0
    assert mixed > 0


def test_score_delivery_variance_needs_word_gaps() -> None:
    from clippos.pipeline.transcribe import build_transcript_timeline

    jitter_timeline = build_transcript_timeline(
        {
            "segments": [
                {
                    "speaker": "a",
                    "start_seconds": 0.0,
                    "end_seconds": 4.0,
                    "text": "one two three four",
                    "words": [
                        {
                            "text": "one",
                            "start_seconds": 0.0,
                            "end_seconds": 0.3,
                            "confidence": 0.9,
                        },
                        {
                            "text": "two",
                            "start_seconds": 0.8,
                            "end_seconds": 1.1,
                            "confidence": 0.9,
                        },
                        {
                            "text": "three",
                            "start_seconds": 1.2,
                            "end_seconds": 1.6,
                            "confidence": 0.9,
                        },
                        {
                            "text": "four",
                            "start_seconds": 3.0,
                            "end_seconds": 3.4,
                            "confidence": 0.9,
                        },
                    ],
                }
            ]
        }
    )

    assert score_delivery_variance(tuple(jitter_timeline.segments)) > 0


def test_has_buried_lead_detects_phrases() -> None:
    assert has_buried_lead("This only makes sense if you saw the earlier part.")
    assert not has_buried_lead("Here's what happened next.")


def test_has_dangling_question_flags_unanswered_tail(long_transcript_timeline) -> None:
    from clippos.pipeline.transcribe import build_transcript_timeline

    dangling = build_transcript_timeline(
        {
            "segments": [
                {
                    "speaker": "a",
                    "start_seconds": 0.0,
                    "end_seconds": 2.0,
                    "text": "We tried the new recipe.",
                    "words": [],
                },
                {
                    "speaker": "a",
                    "start_seconds": 2.0,
                    "end_seconds": 4.0,
                    "text": "Was it worth it?",
                    "words": [],
                },
            ]
        }
    )

    assert has_dangling_question(tuple(dangling.segments)) is True
    assert has_dangling_question(tuple(long_transcript_timeline.segments)) is False


def test_is_rambling_middle_flags_low_motion_low_keyword(
    long_transcript_timeline, long_vision_timeline
) -> None:
    middle_window = tuple(long_transcript_timeline.segments[1:4])

    assert (
        is_rambling_middle(
            middle_window,
            long_vision_timeline.frames,
            motion_ceiling=0.9,
            keyword_floor=0.9,
        )
        is True
    )
    assert (
        is_rambling_middle(
            middle_window,
            long_vision_timeline.frames,
            motion_ceiling=0.0,
            keyword_floor=0.0,
        )
        is False
    )


def test_derive_spike_categories_reflects_signal_mix() -> None:
    confrontation = WindowSignals(
        hook=0.3,
        keyword=0.3,
        numeric=0.0,
        interjection=0.0,
        payoff=0.0,
        question_to_answer=0.0,
        motion=0.3,
        shot_change=0.1,
        face_presence=0.9,
        speaker_interaction=0.2,
        delivery_variance=0.0,
        interview_keyword=0.0,
        buried_lead=False,
        dangling_question=False,
        rambling_middle=False,
    )
    payoff_signals = WindowSignals(
        hook=0.3,
        keyword=0.0,
        numeric=0.3,
        interjection=0.0,
        payoff=0.6,
        question_to_answer=0.0,
        motion=0.3,
        shot_change=0.0,
        face_presence=0.5,
        speaker_interaction=0.0,
        delivery_variance=0.0,
        interview_keyword=0.0,
        buried_lead=False,
        dangling_question=False,
        rambling_middle=False,
    )

    assert "emotional_confrontation" in derive_spike_categories(confrontation)
    assert "unusually_useful_claim" in derive_spike_categories(payoff_signals)


def test_derive_reasons_includes_clarity_fallback() -> None:
    signals = WindowSignals(
        hook=0.0,
        keyword=0.0,
        numeric=0.0,
        interjection=0.0,
        payoff=0.0,
        question_to_answer=0.0,
        motion=0.0,
        shot_change=0.0,
        face_presence=0.0,
        speaker_interaction=0.0,
        delivery_variance=0.0,
        interview_keyword=0.0,
        buried_lead=False,
        dangling_question=False,
        rambling_middle=False,
    )

    assert derive_reasons(signals) == ["clarity"]


def test_generate_candidates_returns_sorted_diverse_clips(
    long_transcript_timeline, long_vision_timeline
) -> None:
    candidates = generate_candidates(
        long_transcript_timeline, long_vision_timeline, max_candidates=3
    )

    assert len(candidates) > 0
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= c.score <= 1.0 for c in candidates)
    assert candidates[0].clip_id == "clip-000"


def test_generate_candidates_honors_duration_policy(
    long_transcript_timeline, long_vision_timeline
) -> None:
    candidates = generate_candidates(
        long_transcript_timeline,
        long_vision_timeline,
        max_candidates=6,
    )

    for candidate in candidates:
        duration = candidate.end_seconds - candidate.start_seconds
        assert 15.0 <= duration <= 45.0


def test_generate_candidates_deduplicates_overlapping_windows(
    long_transcript_timeline, long_vision_timeline
) -> None:
    candidates = generate_candidates(
        long_transcript_timeline,
        long_vision_timeline,
        max_candidates=10,
        config=MiningConfig(score_floor=0.0, min_candidates=0, max_overlap_ratio=0.3),
    )

    for index, earlier in enumerate(candidates):
        for later in candidates[index + 1 :]:
            overlap = max(
                0.0,
                min(earlier.end_seconds, later.end_seconds)
                - max(earlier.start_seconds, later.start_seconds),
            )
            shorter = min(
                earlier.end_seconds - earlier.start_seconds,
                later.end_seconds - later.start_seconds,
            )
            if shorter > 0:
                assert overlap / shorter <= 0.3 + 1e-9


def test_generate_candidates_penalizes_buried_lead_windows(
    long_vision_timeline,
) -> None:
    hook_timeline = build_transcript_timeline(
        {
            "segments": [
                {
                    "speaker": "a",
                    "start_seconds": 0.0,
                    "end_seconds": 8.0,
                    "text": "Here's the thing: this crazy secret lands hard.",
                    "words": [],
                },
                {
                    "speaker": "a",
                    "start_seconds": 8.0,
                    "end_seconds": 16.0,
                    "text": "Turns out the outcome was unreal and the room went wild.",
                    "words": [],
                },
            ]
        }
    )
    buried_timeline = build_transcript_timeline(
        {
            "segments": [
                {
                    "speaker": "a",
                    "start_seconds": 0.0,
                    "end_seconds": 8.0,
                    "text": "This only makes sense if you watched the previous ten minutes.",
                    "words": [],
                },
                {
                    "speaker": "a",
                    "start_seconds": 8.0,
                    "end_seconds": 16.0,
                    "text": "Turns out the outcome was unreal and the room went wild.",
                    "words": [],
                },
            ]
        }
    )

    hook_candidates = generate_candidates(
        hook_timeline,
        long_vision_timeline,
        max_candidates=1,
        config=MiningConfig(score_floor=0.0),
    )
    buried_candidates = generate_candidates(
        buried_timeline,
        long_vision_timeline,
        max_candidates=1,
        config=MiningConfig(score_floor=0.0),
    )

    assert hook_candidates and buried_candidates
    assert hook_candidates[0].score > buried_candidates[0].score


def test_generate_candidates_returns_empty_when_max_candidates_is_zero(
    long_transcript_timeline, long_vision_timeline
) -> None:
    assert (
        generate_candidates(
            long_transcript_timeline, long_vision_timeline, max_candidates=0
        )
        == []
    )


def test_generate_candidates_skips_below_score_floor(
    long_transcript_timeline, long_vision_timeline
) -> None:
    strict = generate_candidates(
        long_transcript_timeline,
        long_vision_timeline,
        max_candidates=10,
        config=MiningConfig(score_floor=0.99, min_candidates=0),
    )

    assert strict == []


def test_generate_candidates_fills_minimum_from_next_best_windows(
    long_transcript_timeline, long_vision_timeline
) -> None:
    candidates = generate_candidates(
        long_transcript_timeline,
        long_vision_timeline,
        max_candidates=5,
        config=MiningConfig(score_floor=0.99, min_candidates=5),
    )

    assert len(candidates) == 5
    assert [candidate.clip_id for candidate in candidates] == [
        "clip-000",
        "clip-001",
        "clip-002",
        "clip-003",
        "clip-004",
    ]


# ---------- M1: speaker_interaction weight bump ----------


def test_speaker_interaction_weight_now_at_parity_with_keyword() -> None:
    """M1 (docs/miner-quality.md): the weight bumped 0.05 -> 0.12 so
    multi-speaker exchanges have parity with hook/keyword/payoff. A
    regression that re-narrows the gap would silently re-create the
    dogfood failure where interview content scored too low to make
    the floor."""
    weights = ScoringWeights()
    assert weights.speaker_interaction == 0.12
    assert weights.speaker_interaction == weights.keyword
    assert weights.speaker_interaction == weights.payoff


# ---------- M3: interview keyword bucket ----------


def test_interview_keyword_spike_fires_on_endorsement_phrases() -> None:
    """The interview keyword bucket exists to surface guest stock-pick /
    endorsement moments that monologue keywords miss. Verify some real
    phrases from the dogfood video would fire it."""
    bloom_endorsement = (
        "I think if you want to follow the playbook, Bloom Energy "
        "is hands down one of the best investments right now."
    )
    monologue_buzz = (
        "This is wild. The whole thing is crazy and absolutely insane."
    )

    interview_score = score_interview_keyword_spike(bloom_endorsement)
    monologue_score = score_interview_keyword_spike(monologue_buzz)

    assert interview_score > 0.4, (
        "interview-tuned phrases must score; this is the dogfood failure mode"
    )
    assert monologue_score == 0.0, (
        "pure monologue buzzwords must NOT trigger interview-keyword "
        "scoring — that would pollute solo-content scoring"
    )


def test_interview_keyword_signal_is_gated_on_speaker_interaction() -> None:
    """Even if the text contains interview phrases, the contribution
    is gated on speaker_interaction >= 0.1 so a solo monologue saying
    "hands down the best" doesn't get the multi-speaker bonus."""
    # Build two transcripts with identical text but different speaker mix.
    interview_text = (
        "If you want to follow the playbook, Bloom Energy is hands down "
        "one of the best long ideas — my biggest position for two years now."
    )
    interview_segments = build_transcript_timeline(
        {
            "segments": [
                {
                    "speaker": "host",
                    "start_seconds": 0.0,
                    "end_seconds": 8.0,
                    "text": "What's your highest conviction idea right now?",
                    "words": [],
                },
                {
                    "speaker": "guest",
                    "start_seconds": 8.0,
                    "end_seconds": 30.0,
                    "text": interview_text,
                    "words": [],
                },
                {
                    "speaker": "host",
                    "start_seconds": 30.0,
                    "end_seconds": 36.0,
                    "text": "Tell me more about the thesis there.",
                    "words": [],
                },
            ]
        }
    )
    monologue_segments = build_transcript_timeline(
        {
            "segments": [
                {
                    "speaker": "host",
                    "start_seconds": 0.0,
                    "end_seconds": 8.0,
                    "text": "Let me share my highest conviction idea right now.",
                    "words": [],
                },
                {
                    "speaker": "host",
                    "start_seconds": 8.0,
                    "end_seconds": 30.0,
                    "text": interview_text,
                    "words": [],
                },
                {
                    "speaker": "host",
                    "start_seconds": 30.0,
                    "end_seconds": 36.0,
                    "text": "I'll explain the thesis next.",
                    "words": [],
                },
            ]
        }
    )
    vision_timeline = build_vision_timeline({"frames": []})

    interview_candidates = generate_candidates(
        interview_segments, vision_timeline, max_candidates=5,
        config=MiningConfig(score_floor=0.0, min_candidates=0),
    )
    monologue_candidates = generate_candidates(
        monologue_segments, vision_timeline, max_candidates=5,
        config=MiningConfig(score_floor=0.0, min_candidates=0),
    )

    # The interview version should outscore the monologue version since
    # the interview-keyword signal is gated on speaker_interaction.
    interview_top_score = max(c.score for c in interview_candidates)
    monologue_top_score = max(c.score for c in monologue_candidates)
    assert interview_top_score > monologue_top_score, (
        f"interview ({interview_top_score:.3f}) should outscore monologue "
        f"({monologue_top_score:.3f}) when interview keywords + multi-speaker "
        "activity coincide"
    )


# ---------- M2: interview-block detection + emission guarantee ----------


def test_detect_interview_blocks_finds_multi_speaker_stretches() -> None:
    """The block detector identifies contiguous transcript stretches with
    real multi-speaker activity so the M2 representation guarantee can
    target them. Single-speaker stretches must NOT be flagged."""
    segments = build_transcript_timeline(
        {
            "segments": [
                # 0-60s: host-only monologue
                {"speaker": "host", "start_seconds": 0.0, "end_seconds": 30.0,
                 "text": "Long monologue about market conditions.", "words": []},
                {"speaker": "host", "start_seconds": 30.0, "end_seconds": 60.0,
                 "text": "More monologue here.", "words": []},
                # 60-120s: alternating host + guest (interview block)
                {"speaker": "guest", "start_seconds": 60.0, "end_seconds": 75.0,
                 "text": "I'm a massive Bloom Energy holder.", "words": []},
                {"speaker": "host", "start_seconds": 75.0, "end_seconds": 80.0,
                 "text": "Why?", "words": []},
                {"speaker": "guest", "start_seconds": 80.0, "end_seconds": 110.0,
                 "text": "Hands down one of the best long ideas right now.", "words": []},
                {"speaker": "host", "start_seconds": 110.0, "end_seconds": 120.0,
                 "text": "Tell me more.", "words": []},
                # 120-180s: host-only outro
                {"speaker": "host", "start_seconds": 120.0, "end_seconds": 180.0,
                 "text": "Closing thoughts from the host alone.", "words": []},
            ]
        }
    )

    blocks = _detect_interview_blocks(
        segments.segments,
        min_duration_seconds=30.0,
        min_transitions=2,
    )

    # Exactly one block — the middle alternating stretch.
    assert len(blocks) == 1
    block = blocks[0]
    assert block.start_seconds < 75.0  # captures host->guest entry turn
    assert block.end_seconds > 110.0   # captures guest->host exit turn
    assert {"host", "guest"} <= block.speakers
    assert block.transition_count >= 2


def test_detect_interview_blocks_ignores_solo_monologue() -> None:
    segments = build_transcript_timeline(
        {
            "segments": [
                {"speaker": "host", "start_seconds": 0.0, "end_seconds": 60.0,
                 "text": "All host all the time.", "words": []},
                {"speaker": "host", "start_seconds": 60.0, "end_seconds": 120.0,
                 "text": "Still just the host.", "words": []},
            ]
        }
    )
    blocks = _detect_interview_blocks(
        segments.segments, min_duration_seconds=30.0, min_transitions=2
    )
    assert blocks == []


def test_interview_block_gets_a_candidate_even_when_below_regular_floor() -> None:
    """The dogfood smoking gun: an 8-min guest interview block produced
    zero candidates because every window inside it scored below 0.35.
    M2 + M5 fix this: the multi_speaker_score_floor (0.20) lets a
    representative window from each block survive selection.

    Construct a transcript where a multi-speaker block scores low (no
    monologue keywords, no interview keywords, just neutral chat) and
    verify the block still produces at least one candidate.
    """
    segments = build_transcript_timeline(
        {
            "segments": [
                # 0-50s: high-scoring monologue with keywords (above floor)
                {"speaker": "host", "start_seconds": 0.0, "end_seconds": 25.0,
                 "text": "Here's the thing — this crazy insane secret will "
                         "blow your mind.", "words": []},
                {"speaker": "host", "start_seconds": 25.0, "end_seconds": 50.0,
                 "text": "The truth is wild and the result is bonkers.", "words": []},
                # 60-130s: interview block, neutral chat (below regular floor)
                {"speaker": "guest", "start_seconds": 60.0, "end_seconds": 75.0,
                 "text": "Hi. Good to see you. Thanks for having me on.", "words": []},
                {"speaker": "host", "start_seconds": 75.0, "end_seconds": 90.0,
                 "text": "Of course. Let's get into it.", "words": []},
                {"speaker": "guest", "start_seconds": 90.0, "end_seconds": 110.0,
                 "text": "Things have been pretty steady this year overall.", "words": []},
                {"speaker": "host", "start_seconds": 110.0, "end_seconds": 130.0,
                 "text": "Yeah I would agree. The vibe has been chill.", "words": []},
            ]
        }
    )
    vision_timeline = build_vision_timeline({"frames": []})

    candidates = generate_candidates(
        segments,
        vision_timeline,
        max_candidates=10,
        # Use the regular floor (0.35) — no min_candidates backfill.
        # The interview block must surface via M2/M5, not via backfill.
        config=MiningConfig(min_candidates=0),
    )

    # At least one candidate must overlap the interview block (60-130s).
    interview_block_candidates = [
        c for c in candidates
        if c.end_seconds > 60.0 and c.start_seconds < 130.0
    ]
    assert len(interview_block_candidates) >= 1, (
        "M2 representation guarantee failed: no candidate from the "
        "interview block survived selection. This is the exact dogfood "
        "regression we shipped M2 to fix."
    )


# ---------- M4: new spike categories ----------


def test_derive_spike_categories_emits_expert_endorsement_on_high_interview_keyword() -> None:
    signals = WindowSignals(
        hook=0.0, keyword=0.0, numeric=0.0, interjection=0.0,
        payoff=0.0, question_to_answer=0.0, motion=0.0,
        shot_change=0.0, face_presence=0.0,
        speaker_interaction=0.5,
        delivery_variance=0.0,
        interview_keyword=0.5,  # strong endorsement signal
        buried_lead=False, dangling_question=False, rambling_middle=False,
    )
    categories = derive_spike_categories(signals)
    assert "expert_endorsement" in categories


def test_derive_spike_categories_emits_specific_pick_when_numeric_and_interview_keyword_coincide() -> None:
    signals = WindowSignals(
        hook=0.0, keyword=0.0, numeric=0.3, interjection=0.0,
        payoff=0.0, question_to_answer=0.0, motion=0.0,
        shot_change=0.0, face_presence=0.0,
        speaker_interaction=0.4,
        delivery_variance=0.0,
        interview_keyword=0.2,
        buried_lead=False, dangling_question=False, rambling_middle=False,
    )
    categories = derive_spike_categories(signals)
    assert "specific_pick" in categories


def test_derive_spike_categories_emits_big_number_on_high_numeric_density() -> None:
    """The dogfood video's clip-003 led with '$100B in a day' but
    big_number never fired because the rubric didn't have it. Now it
    fires on raw numeric density independent of speaker mix — concrete
    quantitative hooks are clip-worthy in any context."""
    signals = WindowSignals(
        hook=0.0, keyword=0.0, numeric=0.6, interjection=0.0,
        payoff=0.0, question_to_answer=0.0, motion=0.0,
        shot_change=0.0, face_presence=0.0,
        speaker_interaction=0.0, delivery_variance=0.0,
        interview_keyword=0.0,
        buried_lead=False, dangling_question=False, rambling_middle=False,
    )
    categories = derive_spike_categories(signals)
    assert "big_number" in categories


# ---------- M5: multi-speaker score floor ----------


def test_mining_config_exposes_multi_speaker_score_floor_below_regular_floor() -> None:
    """Sanity check the relaxed floor is actually relaxed. If a future
    edit accidentally sets them equal or inverts them, the M2
    representation guarantee silently stops doing its job."""
    cfg = MiningConfig()
    assert cfg.multi_speaker_score_floor < cfg.score_floor
    assert cfg.multi_speaker_score_floor >= 0.0
