# Render Manifest Contract

`RenderManifest` is the per-clip handoff that ties an approved candidate to the artifacts produced by the render stage (FFmpeg + ASS captions). It is written to disk at `<workspace>/renders/<clip_id>/render-manifest.json` as soon as the FFmpeg command completes, so that downstream tools (QA, publishing) can trace every output back to the source job without re-running the pipeline.

## Shape

A `RenderManifest` contains:

- `clip_id` — matches the candidate identifier carried from mining through review.
- `approved` — boolean gate; renders are skipped when false (M1.6 will wire in human approval; today the stage is driven explicitly by `--stage render`).
- `source_video` — absolute `Path` to the original media the clip was cut from.
- `start_seconds` / `end_seconds` — clip window, in source-timeline seconds. Validated to be strictly increasing.
- `outputs` — `dict[AspectRatio, Path]` mapping each rendered ratio to its MP4 output path.
- `crop_plans` — `dict[AspectRatio, CropPlan]` with one plan per rendered ratio. The set of keys must match `outputs` exactly, and each plan's `aspect_ratio` must equal its key.
- `caption_plan` — ordered `list[CaptionLine]` covering the clip window, retimed relative to `start_seconds`.

### `CropPlan`

- `aspect_ratio` — one of `9:16`, `1:1`, `16:9`.
- `source_width` / `source_height` — pinned from the source probe (positive ints).
- `target_width` / `target_height` — even pixel dims ≤ source dims, chosen to preserve as much resolution as the aspect ratio allows.
- `anchors` — non-empty `list[CropAnchor]`. Each anchor has `timestamp_seconds` (relative to clip start) and `center_x` / `center_y` in `[0, 1]`. Anchors are produced by a OneEuro filter over face centers (falling back to `(0.5, 0.5)` when no face is visible) and are used to compute the static crop origin fed to FFmpeg.

### `CaptionLine` / `CaptionWord`

- `CaptionLine.start_seconds` / `end_seconds` are clip-relative and strictly increasing.
- `CaptionLine.words` is the word-level timeline, each `CaptionWord` carrying `text`, clip-relative `start_seconds` / `end_seconds`, and an `emphasis` flag.
- Emphasis is set when a word contains a digit, is fully upper-case (length > 1), or passes the length / stopword filters in `pipeline/captions.py`.
- Words that straddle the clip boundary are clipped to the window; words with no overlap are dropped.

## Output layout

The render stage writes everything under the job workspace (`<output_dir>/jobs/<job_id>/`):

```
workspace/
  review-manifest.json          # input: the approved clip list
  renders/
    <clip_id>/
      render-manifest.json      # one per clip
      <clip_id>-9x16.mp4
      <clip_id>-1x1.mp4
      <clip_id>-16x9.mp4
      <clip_id>-9x16.ass        # ASS subtitle sidecars (per ratio)
      <clip_id>-1x1.ass
      <clip_id>-16x9.ass
  render-report.json            # summary index across all clips
```

`render-report.json` is the single artifact returned by `stage="render"`. It carries `job_id`, the canonical `video_path`, and a `clips` array where each entry lists the clip's `manifest_path`, per-ratio `outputs`, and `render_dir` (all paths are workspace-relative when they live inside it).

Canonical output dimensions are fixed at render time:

- `9:16` → 1080×1920
- `1:1` → 1080×1080
- `16:9` → 1920×1080

The crop step picks the largest aspect-correct region of the source frame, and FFmpeg scales that region to the canonical dim with `flags=lanczos`. This keeps a single pixel budget regardless of source resolution and matches what social platforms expect.

## FFmpeg contract

Each ratio is rendered with a single FFmpeg invocation built by `adapters/ffmpeg_render.build_ffmpeg_command(...)`:

- `-ss <start> -to <end>` bounds bracket the input, so only the clip window is decoded.
- Video filter chain: `crop=W:H:x:y,scale=CANONICAL_W:CANONICAL_H:flags=lanczos,ass='<subtitle path>'`.
- Codec flags: `libx264 -preset medium -crf 18 -pix_fmt yuv420p -movflags +faststart`, AAC audio at 192 kbps stereo.

The crop origin is the clamped centroid of the OneEuro-smoothed anchors — we keep the crop static within a clip for predictable composition and run Lanczos scaling instead of animated crop expressions. Animated crops can be layered on later without changing the manifest shape.

ASS subtitle sidecars are written next to the MP4 by default (overridable via `subtitle_dir=`). The style defaults to bold Helvetica Neue, normal words in white (`&H00FFFFFF&`), emphasis words in yellow (`&H0000F0FF&`), with font size and vertical margin scaled to the canonical play height.

## Validation rules

- `end_seconds > start_seconds`.
- `outputs.keys() == crop_plans.keys()`.
- Every `crop_plans[r].aspect_ratio == r`.
- `CropPlan.target_width <= source_width`, `target_height <= source_height`, and `anchors` non-empty.
- `CaptionLine.end_seconds > start_seconds`; each `CaptionWord.end_seconds >= start_seconds`.
- Aspect-ratio keys outside the `{9:16, 1:1, 16:9}` literal set are rejected by Pydantic.

## Producer / consumer

- **Producer**: `pipeline.render.build_render_plan(...)` assembles the manifest from a `CandidateClip`, `TranscriptTimeline`, `VisionTimeline`, `MediaProbe`, and the job workspace. The render stage in the orchestrator calls this per approved candidate, hands the manifest to `adapters.ffmpeg_render.render_clip(...)`, then persists the manifest to disk.
- **Consumer**: future publishing / QA steps read the manifest to map source video + clip bounds to a specific set of MP4s and caption tracks. `render-report.json` is the convenient entry point when operating across an entire job.

## Not yet covered

- Animated crop interpolation (kept off for M1.5 quality-vs-simplicity tradeoff; the OneEuro anchors are already stored so we can switch later).
- Retake / regeneration semantics (will land with the M1.6 approval loop).
- Upload / publishing metadata.
