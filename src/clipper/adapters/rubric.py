from __future__ import annotations

RUBRIC_VERSION = "1.1.0"

RUBRIC_PROMPT = """\
You are scoring short video clips pulled from a longer recording. Each clip in
`clips` is a candidate for standalone short-form distribution. You must score
every clip in the input and return strict JSON matching `response_schema`.

## Rubric dimensions (each 0.0–1.0)

- hook: strength of the first 1–3 seconds. Does the opening grab attention
  instantly, or does it need setup before it lands?
- shareability: would a typical viewer forward this to a friend, quote it, or
  screenshot a line? Does the clip provoke a reaction?
- standalone_clarity: can someone who has never seen the source recording
  follow the clip without confusion? Score low if it references "what I said
  earlier" or unnamed prior context.
- payoff: does the clip deliver on whatever it sets up — a question, a tease,
  an implied stake? Score low if the payoff is deferred or missing.
- delivery_energy: pace, cadence, vocal variety, expressiveness. Flat and
  monotone reads score low; animated, varied delivery scores high.
- quotability: density of memorable, screenshot-worthy, or repeatable lines.
  One-liners, sharp framings, and vivid phrasings raise this.

Be honest and granular. Do not default to 0.5. Use the full range.

## Positive spike categories (include zero or more, only if clearly present)

- emotional_confrontation: visible conflict, heated tone, real stakes between
  speakers.
- controversy: the clip takes a clear side on a charged or disputed issue.
- taboo: discusses something forbidden, hidden, secret, or socially unspoken.
- absurdity: surreal, bizarre, or unexpectedly weird content.
- action: physical action, visible event, strong reaction shots.
- unusually_useful_claim: a novel, crisp, actionable insight or framework —
  something a viewer could apply.
- expert_endorsement: a credible guest or expert names a specific person,
  company, asset, or strategy as exceptional ("hands down the best",
  "one of the smartest people I know"). Multi-speaker context expected.
- specific_pick: a guest or host calls out a specific stock, ticker, asset,
  or trade — usually with a position word ("I'm long $X", "the play here is",
  "my biggest position"). Common in finance / podcast verticals.
- big_number: the clip leads on or pivots around a striking quantitative
  claim ("$100B in a day", "lost 80% in two weeks", "10x in three months").
  Concrete numbers that read at-a-glance, not vague magnitudes.

Do not invent new categories. Only use categories listed above.

## Penalties (include zero or more, only if clearly present)

- buried_lead: the strongest moment sits far from the start of the clip.
- dangling_question: the clip sets up a question but never answers it.
- rambling_middle: energy dips in the middle, filler or losing thread.
- context_dependent: requires outside knowledge from earlier in the recording.
- low_delivery: flat, monotone, unclear, or low-energy throughout.

Do not invent new penalties. Only use penalties listed above.

## Output fields (per clip)

- clip_id: echo the `clip_id` you received for this clip.
- clip_hash: echo the `clip_hash` you received for this clip.
- rubric: an object with all six rubric dimensions (0.0–1.0).
- spike_categories: array of zero or more spike category strings.
- penalties: array of zero or more penalty strings.
- final_score (0.0–1.0): your overall quality judgment — bias low when
  uncertain; reserve high scores for clips you would actually publish.
- title: short, specific, editorial headline. Max 60 characters. No clickbait.
- hook: the opening line rewritten as an attention grabber. Max 90 characters.
- reasons: 2–4 terse phrases explaining the score in plain language.

Return every clip from the input. Do not drop any. Do not add any.
"""


def build_rubric_prompt() -> str:
    return RUBRIC_PROMPT


def build_response_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["rubric_version", "job_id", "scores"],
        "properties": {
            "rubric_version": {"type": "string", "const": RUBRIC_VERSION},
            "job_id": {"type": "string"},
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "clip_id",
                        "clip_hash",
                        "rubric",
                        "spike_categories",
                        "penalties",
                        "final_score",
                        "title",
                        "hook",
                        "reasons",
                    ],
                    "properties": {
                        "clip_id": {"type": "string"},
                        "clip_hash": {"type": "string"},
                        "rubric": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "hook",
                                "shareability",
                                "standalone_clarity",
                                "payoff",
                                "delivery_energy",
                                "quotability",
                            ],
                            "properties": {
                                "hook": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "shareability": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "standalone_clarity": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "payoff": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "delivery_energy": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "quotability": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                            },
                        },
                        "spike_categories": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "emotional_confrontation",
                                    "controversy",
                                    "taboo",
                                    "absurdity",
                                    "action",
                                    "unusually_useful_claim",
                                    "expert_endorsement",
                                    "specific_pick",
                                    "big_number",
                                ],
                            },
                        },
                        "penalties": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "buried_lead",
                                    "dangling_question",
                                    "rambling_middle",
                                    "context_dependent",
                                    "low_delivery",
                                ],
                            },
                        },
                        "final_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "title": {"type": "string"},
                        "hook": {"type": "string"},
                        "reasons": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
        },
    }
