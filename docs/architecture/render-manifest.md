# Render Manifest Contract

`RenderManifest` is the narrow handoff between clip review approval and future render execution. The current implementation keeps this contract intentionally small: approval state plus aspect-ratio keyed output paths.

## Shape

The manifest currently contains:

- `clip_id`: the reviewed candidate identifier.
- `approved`: whether the clip passed review gating.
- `outputs`: a mapping of constrained aspect ratio keys to planned `Path` values.

The allowed output keys are fixed to:

- `9:16`
- `1:1`
- `16:9`

`build_render_plan(...)` currently produces clip-local filenames:

- `{clip_id}-9x16.mp4`
- `{clip_id}-1x1.mp4`
- `{clip_id}-16x9.mp4`

These are planning artifacts only. They describe expected render outputs and preserve the current model constraint that every output value is a `Path`.

## Planning Helpers

The render module currently also exposes two simple helpers:

- `build_caption_lines(transcript_segment)` returns a single caption line containing the segment text and marks the first two transcript words for emphasis.
- `choose_crop_anchor(frame)` returns the normalized `primary_face.center_x` and `primary_face.center_y` when a face is present, otherwise it falls back to `(0.5, 0.5)`.

The crop helper stays deterministic because the upstream vision timeline already normalizes `primary_face` coordinates to the `[0, 1]` range.

## Scope Boundaries

This contract and module do not yet include:

- FFmpeg arguments
- output directories
- timeline-based crop animation
- multi-line caption segmentation
- actual render execution state

Those concerns should be layered onto the manifest later without weakening the current aspect-ratio and path constraints.
