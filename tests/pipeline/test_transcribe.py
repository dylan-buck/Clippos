import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from clipper.pipeline.transcribe import build_transcript_timeline


@pytest.fixture
def sample_transcript_payload() -> dict:
    fixture_path = (
        Path(__file__).resolve().parents[1] / "fixtures" / "sample_transcript.json"
    )
    return json.loads(fixture_path.read_text())


def test_build_transcript_timeline_returns_word_ranges(
    sample_transcript_payload: dict,
) -> None:
    timeline = build_transcript_timeline(sample_transcript_payload)

    assert timeline.segments[0].speaker == "speaker_1"
    assert timeline.segments[0].start_seconds == 0.0
    assert timeline.segments[0].words[0].text == "Look"
    assert timeline.segments[0].words[0].start_seconds == 0.0
    assert timeline.segments[0].words[0].end_seconds == 0.28


def test_build_transcript_timeline_keeps_speaker_turns(
    sample_transcript_payload: dict,
) -> None:
    timeline = build_transcript_timeline(sample_transcript_payload)

    assert [segment.speaker for segment in timeline.segments] == [
        "speaker_1",
        "speaker_2",
    ]


def test_build_transcript_timeline_allows_zero_length_word_and_segment() -> None:
    timeline = build_transcript_timeline(
        {
            "segments": [
                {
                    "speaker": "speaker_1",
                    "start_seconds": 0.0,
                    "end_seconds": 0.0,
                    "text": "Look",
                    "words": [
                        {
                            "text": "Look",
                            "start_seconds": 0.0,
                            "end_seconds": 0.0,
                            "confidence": 0.99,
                        }
                    ],
                }
            ]
        }
    )

    assert timeline.segments[0].end_seconds == 0.0
    assert timeline.segments[0].words[0].end_seconds == 0.0


def test_build_transcript_timeline_rejects_reversed_word_time_bounds(
    sample_transcript_payload: dict,
) -> None:
    sample_transcript_payload["segments"][0]["words"][0]["start_seconds"] = 0.5
    sample_transcript_payload["segments"][0]["words"][0]["end_seconds"] = 0.2

    with pytest.raises(ValidationError):
        build_transcript_timeline(sample_transcript_payload)


def test_build_transcript_timeline_rejects_negative_segment_times(
    sample_transcript_payload: dict,
) -> None:
    sample_transcript_payload["segments"][1]["end_seconds"] = -0.1

    with pytest.raises(ValidationError):
        build_transcript_timeline(sample_transcript_payload)


def test_build_transcript_timeline_rejects_reversed_segment_time_bounds(
    sample_transcript_payload: dict,
) -> None:
    sample_transcript_payload["segments"][1]["start_seconds"] = 3.1
    sample_transcript_payload["segments"][1]["end_seconds"] = 3.0

    with pytest.raises(ValidationError):
        build_transcript_timeline(sample_transcript_payload)


def test_build_transcript_timeline_rejects_malformed_payload_shape() -> None:
    with pytest.raises(ValidationError):
        build_transcript_timeline({"segments": [{"speaker": "speaker_1"}]})
