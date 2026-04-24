# Job Spec

`ClipperJob` is the stable handoff contract for the current local pipeline. It
defines the source video, destination directory, review gate, output profile, and
candidate cap used by ingest and review-manifest generation. All contract models
reject unexpected fields to keep handoffs explicit and stable.

## Contract Summary

- `video_path`: absolute or workspace-resolved source media path
- `output_dir`: destination directory for per-job workspace artifacts
- `review_required`: gate that defaults to `true`
- `output_profile.ratios`: output aspect ratios rendered during `stage=render`
  in stable priority order
- `output_profile.caption_preset`: caption style preset resolved at render time.
  Must be one of `hook-default`, `bottom-creator`, `bottom-compact`,
  `lower-third-clean`, `center-punch`, `top-clean`. Defaults to `hook-default`
  (alias of `bottom-creator`). See
  [render-manifest.md](render-manifest.md#caption-presets) for the catalog.
- `max_candidates`: maximum number of clips surfaced for review

The CLI currently validates a JSON job payload with this contract before invoking
`run_job(...)`.

## Pipeline Stages

`run_job(job, *, stage=...)` supports four stages:

- `mine` — ingest, transcribe, analyze vision, and write
  `scoring-request.json`; exits at the mine boundary.
- `review` — load the existing scoring request plus a matching
  `scoring-response.json` (or per-clip cache) and write `review-manifest.json`.
- `render` — load `review-manifest.json`, re-derive transcript / vision
  timelines, build a per-clip `RenderManifest` for candidates marked
  `approved: true`, shell out to FFmpeg to produce 9:16 / 1:1 / 16:9 MP4s with
  ASS caption sidecars, and emit `render-report.json`. The stage exits with a
  `RenderStageError` when no candidates are approved.
- `auto` (default) — run `mine`, then continue into `review` when scores are
  already resolved; otherwise stop after emitting the request. `auto` does not
  chain into render — the render stage must be invoked explicitly after the
  review manifest has been approved.

Workspace artifacts all live under `<output_dir>/jobs/<job_id>/`:
`scoring-request.json`, `scoring-response.json`, `scoring-cache/*.json`,
`review-manifest.json`, `renders/<clip_id>/*`, `render-report.json`, and —
once the `/clip-package` flow runs — `package-request.json`,
`package-response.json`, `package-report.json`, plus per-clip
`renders/<clip_id>/package.json`. See
[scoring-handoff.md](scoring-handoff.md) for the scoring handoff contract,
[render-manifest.md](render-manifest.md) for the render stage contract, and
[package-handoff.md](package-handoff.md) for the packaging handoff contract.

The agent skill layer uses this same contract. `/clip` defaults to all three
ratios because rendering does not use model calls, but narrower user requests
are written into `output_profile.ratios` before render.

## Validation Rules

- Unknown fields are rejected across the shared contract models
- `max_candidates` must be greater than `0`
- `MediaProbe` numeric fields must all be greater than `0`
- `CandidateClip.start_seconds` and `CandidateClip.end_seconds` must be non-negative
- `CandidateClip.end_seconds` must be greater than `start_seconds`
- `CandidateClip.score` is bounded to `0.0` through `1.0`
- `CandidateClip.approved` defaults to `false`; render exports only approved
  candidates
- Aspect ratios are restricted to the shared vocabulary: `9:16`, `1:1`, `16:9`

## Example

```json
{
  "video_path": "/tmp/input.mp4",
  "output_dir": "/tmp/out",
  "review_required": true,
  "output_profile": {
    "ratios": ["9:16", "1:1", "16:9"],
    "caption_preset": "hook-default"
  },
  "max_candidates": 12
}
```

## Related Shared Models

- `MediaProbe`: normalized probe metadata used by analysis and render steps
- `CandidateClip`: scored clip candidate with time bounds and rationale
- `ReviewManifest`: review payload tying a job and source video to candidates
- `RenderManifest`: per-clip render contract carrying bounds, crop plans, caption plan, and aspect-ratio keyed output paths
