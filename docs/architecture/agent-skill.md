# Agent Skill Workflow

Clippos is packaged as an agent skill so users can invoke the engine
with `/clippos` instead of manually running every CLI stage.

## Commands

- `/clippos <video>` runs the full local workflow for a video path, attached video
  file, direct video URL, or platform URL when `yt-dlp` is installed.
- `/clippos-config` checks or writes local defaults such as output directory,
  ratios, candidate count, approval threshold, and Hugging Face token status.
- `/clippos-package` generates titles, thumbnail overlay lines, social captions,
  hashtags, and hooks for already-rendered clips.

Command shims live in `commands/`. The skill contract lives in root `SKILL.md`.

## Runtime Split

The skill keeps expensive model usage inside the hosting harness:

- Local engine: probe, transcription, diarization, vision analysis, candidate
  mining, review manifest generation, crop planning, caption planning, FFmpeg
  rendering.
- Harness model: brief authoring from `brief-request.json`, semantic scoring of
  `scoring-request.json` into `scoring-response.json`, and optional packaging
  metadata authoring.
- Skill helper: config loading, source preparation, approval selection, and
  final output reporting.

## Helper Script

`scripts/clippos_skill.py` provides deterministic operations that agents should
not reimplement. Resolve the skill root against `CLAUDE_PLUGIN_ROOT` when
installed as a plugin, otherwise the repo checkout:

```bash
CLIPPOS_ROOT="${CLIPPOS_ROOT:-${CLAUDE_PLUGIN_ROOT:-$PWD}}"
CLIPPOS_PYTHON="${CLIPPOS_PYTHON:-$CLIPPOS_ROOT/.venv/bin/python}"
[ -x "$CLIPPOS_PYTHON" ] || CLIPPOS_PYTHON="$(command -v python3)"
"$CLIPPOS_PYTHON" "$CLIPPOS_ROOT/scripts/clippos_skill.py" config-check
"$CLIPPOS_PYTHON" "$CLIPPOS_ROOT/scripts/clippos_skill.py" config-write --output-dir ~/Documents/Clippos
"$CLIPPOS_PYTHON" "$CLIPPOS_ROOT/scripts/clippos_skill.py" prepare "$SOURCE" --ratios "9:16,1:1"
"$CLIPPOS_PYTHON" "$CLIPPOS_ROOT/scripts/clippos_skill.py" approve "$REVIEW_MANIFEST" --top 3 --min-score 0.70
"$CLIPPOS_PYTHON" "$CLIPPOS_ROOT/scripts/clippos_skill.py" outputs "$RENDER_REPORT"
```

The helper reads `~/.config/clippos/.env`, then lets real environment
variables override file values. URL sources are validated with `ffprobe` at
download time so an HTML error page saved as `.mp4` fails at the boundary
instead of crashing the mine stage.

## Config Keys

- `CLIPPOS_OUTPUT_DIR`: default output root.
- `CLIPPOS_RATIOS`: comma-separated subset of `9:16`, `1:1`, `16:9`; defaults
  to all.
- `CLIPPOS_MAX_CANDIDATES`: candidate count requested from mining.
- `CLIPPOS_APPROVE_TOP`: max clips approved by the skill helper.
- `CLIPPOS_MIN_SCORE`: approval threshold; if no clip clears it, the best clip
  is approved so the user gets output.
- `HF_TOKEN`: optional. Only needed for the pyannote diarization opt-in
  (`CLIPPOS_DIARIZER=pyannote`). Default diarizer is the open-source
  SpeechBrain stack; no token required.

## End-To-End Shape

1. `/clippos` parses source and options.
2. `prepare` writes a job JSON.
3. `clippos.cli run --stage mine` writes `scoring-request.json` and, by
   default, `brief-request.json`.
4. The harness model writes `brief-response.json`; `--stage brief` embeds the
   `video_brief` into `scoring-request.json`.
5. The harness model writes `scoring-response.json`.
6. `clippos.cli run --stage review` writes `review-manifest.json`.
7. `approve` marks selected candidates with `approved: true`.
8. `clippos.cli run --stage render` writes final MP4s and `render-report.json`.
9. `outputs` formats the rendered paths for the final response.

The Hermes-first driver, `scripts/hermes_clippos.py`, wraps these primitives
into a resumable `advance` loop. It also persists per-run approval settings,
auto-approves top clips, renders immediately after scoring, exposes packaging
as a next step, and records keep/skip feedback.

## Output Location

Final MP4s are under:

```text
<output_dir>/jobs/<job_id>/renders/<clip_id>/
```

`render-report.json` is the index for all rendered clips and ratios.
