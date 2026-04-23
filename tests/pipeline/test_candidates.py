from __future__ import annotations

import pytest

from clipper.pipeline.candidates import (
    DurationPolicy,
    MiningConfig,
    WindowSignals,
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
    score_keyword_spike,
    score_motion_density,
    score_numeric_density,
    score_payoff_signal,
    score_question_to_answer,
    score_shot_change_density,
    score_speaker_interaction,
)
from clipper.pipeline.transcribe import build_transcript_timeline
from clipper.pipeline.vision import build_vision_timeline


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
    from clipper.pipeline.transcribe import build_transcript_timeline

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
    from clipper.pipeline.transcribe import build_transcript_timeline

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
    from clipper.pipeline.transcribe import build_transcript_timeline

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
        config=MiningConfig(score_floor=0.0, max_overlap_ratio=0.3),
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
        config=MiningConfig(score_floor=0.99),
    )

    assert strict == []
