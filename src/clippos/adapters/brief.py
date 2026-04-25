"""Video brief prompt + JSON schema for the v1.1 brief stage.

The brief is the model's pre-scoring synthesis of what makes a given
video clippable. See docs/v1.1.md for the motivating case study.

The prompt is intentionally short and opinionated. It asks the model to
read a transcript excerpt and produce a structured, scoring-relevant
frame, NOT a summary. The schema enforces field presence and length so
downstream scoring/packaging can rely on the shape.
"""
from __future__ import annotations

from clippos.adapters.rubric import RUBRIC_VERSION

# Bumped together with RUBRIC_VERSION since the brief is part of the
# scoring contract. A breaking change to the brief schema invalidates
# cached scores the same way a rubric change does.
BRIEF_VERSION = RUBRIC_VERSION

BRIEF_PROMPT = """\
You are reading a transcript excerpt from a long-form video and authoring a
one-paragraph VideoBrief that will guide per-clip scoring downstream.

Your job is NOT to summarize the video. It is to produce an opinionated,
scoring-relevant frame: what makes a clip from THIS video specifically worth
keeping, distinct from the generic rubric.

A useful brief lets the per-clip scorer answer two questions it cannot
answer alone:

1. What is the spine of this video? Which themes / moments would a thoughtful
   editor build the highlight reel around?
2. What looks clip-worthy on the surface but should be down-weighted because
   it is off-thesis or redundant for this specific source?

Read the transcript excerpt carefully. If the excerpt is truncated, infer
the global shape from what you can see plus the speaker mix and duration.

## Required output fields

- theme: 1-2 sentences naming the central thesis or arc of the video.
  Be specific. "About crypto" is wrong; "host's pivot from active crypto
  trading to AI investing, framed as a personal commitment" is right.
- video_format: short label for the format. Examples: "stream recap,
  single-host", "podcast interview, two speakers", "founder vlog",
  "panel discussion, multi-speaker", "tutorial walkthrough".
- expected_viral_patterns: 3-5 specific kinds of moments a clipper should
  prioritize for THIS video. Tied to the theme. Examples: "the central
  'why I quit' reveal", "specific dollar-amount loss anecdotes",
  "guest's expert stock-pick endorsements".
- anti_patterns: 0-3 specific kinds of moments a clipper should
  down-weight even if they trip generic rubric heuristics. Examples:
  "intro music or sponsor reads", "technical TA jargon without payoff",
  "stream-chat banter that lacks standalone clarity".
- audience: optional, 1 short phrase naming the target viewer.
  ("crypto-curious finance audience", "indie founders").
- tone: optional, 1 short phrase. ("candid, mildly self-deprecating",
  "high-energy, didactic", "conversational, expert-Q&A").
- notes: optional, 1-2 sentences of free-form context that doesn't fit
  the other fields.

Be concrete. Vague briefs ("something that grabs attention") are worse
than no brief because they confuse downstream scoring.

Return strict JSON matching `response_schema`. Echo `rubric_version` and
`job_id` exactly as received.
"""


def build_brief_prompt() -> str:
    return BRIEF_PROMPT


def build_brief_response_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["rubric_version", "job_id", "brief"],
        "properties": {
            "rubric_version": {"type": "string", "const": BRIEF_VERSION},
            "job_id": {"type": "string"},
            "brief": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "rubric_version",
                    "job_id",
                    "theme",
                    "video_format",
                    "expected_viral_patterns",
                    "anti_patterns",
                ],
                "properties": {
                    "rubric_version": {
                        "type": "string",
                        "const": BRIEF_VERSION,
                    },
                    "job_id": {"type": "string"},
                    "theme": {"type": "string", "minLength": 1},
                    "video_format": {"type": "string", "minLength": 1},
                    "expected_viral_patterns": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 5,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "anti_patterns": {
                        "type": "array",
                        "maxItems": 3,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "audience": {"type": ["string", "null"]},
                    "tone": {"type": ["string", "null"]},
                    "notes": {"type": ["string", "null"]},
                },
            },
        },
    }
