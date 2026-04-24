# Render Manifest Contract

`RenderManifest` is the per-clip handoff that ties an approved candidate to the artifacts produced by the render stage (FFmpeg + ASS captions). It is written to disk at `<workspace>/renders/<clip_id>/render-manifest.json` as soon as the FFmpeg command completes, so that downstream tools (QA, publishing) can trace every output back to the source job without re-running the pipeline.

## Shape

A `RenderManifest` contains:

- `clip_id` — matches the candidate identifier carried from mining through review.
- `approved` — boolean gate; renders are skipped when false. The orchestrator
  builds render manifests only for review candidates marked `approved: true`.
- `source_video` — absolute `Path` to the original media the clip was cut from.
- `start_seconds` / `end_seconds` — clip window, in source-timeline seconds. Validated to be strictly increasing.
- `outputs` — `dict[AspectRatio, Path]` mapping each rendered ratio to its MP4 output path.
- `crop_plans` — `dict[AspectRatio, CropPlan]` with one plan per rendered ratio. The set of keys must match `outputs` exactly, and each plan's `aspect_ratio` must equal its key.
- `caption_plan` — ordered `list[CaptionLine]` covering the clip window, retimed relative to `start_seconds`.
- `mode` — `SceneMode` literal (`TRACK` or `GENERAL`), derived from the vision timeline by `pipeline.scene_strategy.derive_clip_mode`. `TRACK` is the default when a primary face is detected in at least half of the clip's vision frames; `GENERAL` kicks in for clips with no face or many transient faces.
- `caption_preset` — `CaptionPreset` literal naming the ASS style catalog entry (see *Caption presets* below). Defaults to `hook-default` (alias of `bottom-creator`). Selected by `ClipperJob.output_profile.caption_preset` and resolved to a `CaptionStyle` at render time by `pipeline.caption_styles.resolve_caption_style`.

### `CropPlan`

- `aspect_ratio` — one of `9:16`, `1:1`, `16:9`.
- `source_width` / `source_height` — pinned from the source probe (positive ints).
- `target_width` / `target_height` — even pixel dims ≤ source dims, chosen to preserve as much resolution as the aspect ratio allows.
- `anchors` — non-empty `list[CropAnchor]`. Each anchor has `timestamp_seconds` (relative to clip start) and `center_x` / `center_y` in `[0, 1]`. Anchors are produced by a "heavy tripod" `SmoothedCameraman` (safe-zone radius with slow/fast pan rates, snap on shot change) and feed the piecewise-linear ffmpeg crop expression in `TRACK` mode. In `GENERAL` mode the anchors are still emitted so the manifest shape stays stable, but the renderer ignores them in favor of a blurred-background composition.

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

Each ratio is rendered with a single FFmpeg invocation built by `adapters/ffmpeg_render.build_ffmpeg_command(...)`. The command dispatches on `RenderManifest.mode`:

- `-ss <start> -to <end>` bounds bracket the input in both modes, so only the clip window is decoded.
- **TRACK mode** — video filter chain: `crop=W:H:x_expr:y_expr,scale=CANONICAL_W:CANONICAL_H:flags=lanczos,ass='<subtitle path>'`. The crop origin is built as a piecewise-linear ffmpeg eval expression over the `SmoothedCameraman` anchors (integer origin pixels, clamped to the source bounds, commas escaped with `\,`). This gives us a virtual camera that holds still within a safe zone and pans at a bounded rate when the subject drifts, snapping on shot changes.
- **GENERAL mode** — `-filter_complex` graph: `[0:v]split=2[bg_src][fg_src]; [bg_src]scale=CW:CH:force_original_aspect_ratio=increase,crop=CW:CH,boxblur=20:1[bg]; [fg_src]scale=CW:CH:force_original_aspect_ratio=decrease[fg]; [bg][fg]overlay=(W-w)/2:(H-h)/2[composited]; [composited]ass='<subs>'[out]`, plus `-map "[out]" -map "0:a?"`. Intended for clips with no clear single subject (landscape, group shots, B-roll): the blurred, cropped background fills the canonical canvas while the original frame sits aspect-preserved in the middle.
- Codec flags (both modes): `libx264 -preset medium -crf 18 -pix_fmt yuv420p -movflags +faststart`, AAC audio at 192 kbps stereo.

ASS subtitle sidecars are written next to the MP4 by default (overridable via `subtitle_dir=`). Font is Helvetica Neue, normal words in white (`&H00FFFFFF&`), emphasis words in yellow (`&H0000F0FF&`). All other style knobs (alignment, bold, font-size ratio, vertical margin, horizontal margins, outline, shadow) come from the resolved `CaptionStyle` — see *Caption presets*.

## Caption presets

`RenderManifest.caption_preset` picks one entry from `pipeline.caption_styles.CAPTION_STYLES`. Each entry is a frozen `CaptionStyle` dataclass:

- `alignment` — ASS numpad grid (2 = bottom-center, 5 = middle-center, 8 = top-center).
- `bold` — maps to the ASS Bold flag (-1/0).
- `font_size_ratio` + `font_size_floor` — font size is `max(round(play_height * ratio), floor)`, so small canvases (1080-tall 1:1 / 16:9) stay legible.
- `margin_v_ratio` + `margin_v_floor` — same clamping for vertical margin.
- `margin_l` / `margin_r` — fixed horizontal margins in pixels.
- `outline` / `shadow` — ASS border / drop-shadow widths.

Shipped catalog:

| Preset | Alignment | Bold | Font (1920 px) | Margin V (1920 px) | Use |
|---|---|---|---|---|---|
| `bottom-creator` | bottom-center | yes | 96 | 230 | Default creator-style bold captions above the bottom edge. |
| `bottom-compact` | bottom-center | yes | 77 | 192 | Slightly smaller footprint for dense word counts. |
| `lower-third-clean` | bottom-center | no | 86 | 422 | Cleaner lower-third placement for interview / podcast clips. |
| `center-punch` | middle-center | yes | 144 | 0 | Big centered punch card for short hooks / reactions. |
| `top-clean` | top-center | no | 86 | 192 | Top-aligned captions for clips where the subject sits low. |

`hook-default` is an alias of `bottom-creator` so existing jobs and cached manifests keep rendering identically after the widen.

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

- Per-scene-within-clip strategy (v1 picks a single `mode` per clip by pooling every vision frame in the clip window; switching between TRACK and GENERAL inside a single clip would require ffmpeg concat plumbing and is deliberately deferred).
- Retake / regeneration semantics.
- Upload / publishing metadata.
