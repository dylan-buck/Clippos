# Scoring Handoff

The clipper delegates clip-quality scoring to the surrounding agent harness
(Claude Code, Codex, or Hermes Agent). The clipper never calls an LLM API on its
own and does not depend on any provider SDK or API key. Instead, it emits a
self-describing scoring request, pauses, and resumes from a scoring response
the harness writes back into the workspace.

## Stages

`run_job(job, *, stage=...)` exposes three stages:

- `mine` — ingest, transcribe, analyze vision, mine candidate windows, write
  `scoring-request.json`, and exit. Returns the request path.
- `review` — load the existing `scoring-request.json`, require resolved scores
  (either from `scoring-response.json` or the per-clip cache), and write
  `review-manifest.json`. Returns the manifest path. Raises
  `ScoringResponseError` when scoring context is missing or incomplete.
- `auto` (default) — run `mine`, then attempt `review` in the same process. If
  resolved scores are not yet available the command stops at the mine stage and
  returns the request path instead of raising.

The CLI mirrors the stage flag: `python -m clipper.cli run job.json --stage mine`.

## Workspace artifacts

All paths are relative to the per-job workspace `<output_dir>/jobs/<job_id>/`.

- `scoring-request.json` — clipper output. Contains `rubric_version`, `job_id`,
  absolute `video_path`, the locked `rubric_prompt`, the JSON `response_schema`,
  and one `ClipBrief` per mined candidate.
- `scoring-response.json` — harness input. Must validate against
  `response_schema` and `ScoringResponse`. One `ClipScore` per brief, keyed by
  `clip_hash`.
- `scoring-cache/<clip_hash>.json` — per-clip cache of validated
  `ClipScore` objects. Populated automatically when a response is merged and
  reused on re-runs so previously scored clips survive re-mining.
- `review-manifest.json` — final review package, produced by the review stage.

## Rubric

The rubric is versioned by `RUBRIC_VERSION` (currently `1.0.0`). Bumping the
version invalidates cached scores and forces a fresh response because
`clip_hash` mixes rubric version with clip bounds and transcript.

Dimensions (each `0.0–1.0`):

- `hook`, `shareability`, `standalone_clarity`, `payoff`, `delivery_energy`,
  `quotability`.

Positive spike categories (zero or more, strict enum):

- `emotional_confrontation`, `controversy`, `taboo`, `absurdity`, `action`,
  `unusually_useful_claim`.

Penalties (zero or more, strict enum):

- `buried_lead`, `dangling_question`, `rambling_middle`, `context_dependent`,
  `low_delivery`.

Per-clip output fields: `clip_id`, `clip_hash`, `rubric`, `spike_categories`,
`penalties`, `final_score`, `title`, `hook`, `reasons`.

## Clip identity and caching

Each brief carries a deterministic `clip_hash`:

```text
sha1("<rubric_version>|<start_seconds:.3f>|<end_seconds:.3f>|<transcript>")[:16]
```

Rules enforced by `resolve_scores(...)`:

- Response `rubric_version` must match the request.
- Response `job_id` must match the request.
- For every brief, the response's matching entry (by `clip_hash`) must echo the
  brief's `clip_id`.
- Any response entry merged successfully is persisted to the per-clip cache.
- When the response omits a brief, the cached `ClipScore` for that `clip_hash`
  is reused. Its `clip_id` is rewritten to the current rank so re-runs with new
  candidate orderings still line up.
- If neither the response nor the cache covers a brief, `resolve_scores`
  returns `None` and `run_job` falls back to emitting the request path.

## Strictness

- `ScoringRequest`, `ClipBrief`, `ScoringResponse`, `ClipScore`, `RubricScores`,
  and `MiningSignals` all use `extra="forbid"`; unknown fields fail validation.
- The embedded `response_schema` also sets `additionalProperties: false`,
  pins `rubric_version` to the current constant, and locks the spike category
  and penalty enums. Harnesses can validate before writing the response.
- Malformed `scoring-response.json` (bad JSON, contract violation, version or
  id mismatch) raises `ScoringResponseError`, which the CLI surfaces with a
  `Scoring handoff error:` prefix and exit code `1`.

## Harness workflow

Each harness wrapper exposes thin helpers over `pipeline.scoring`:

- `{prefix}_load_scoring_request(workspace_dir)` — load and validate the
  request, raising `FileNotFoundError` if absent.
- `{prefix}_write_scoring_response(workspace_dir, response)` — accept either a
  `ScoringResponse` model or a dict payload; the dict form is validated before
  being written.

Typical end-to-end flow:

1. Harness (or CLI) invokes `run_job(job, stage="mine")`.
2. Harness loads `scoring-request.json`, asks its in-session LLM to score every
   clip against the embedded rubric and schema.
3. Harness writes `scoring-response.json` via the wrapper helper.
4. Harness invokes `run_job(job, stage="review")` (or reruns `auto`) to get
   `review-manifest.json`.
