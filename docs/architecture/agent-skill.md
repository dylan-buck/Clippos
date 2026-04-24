# Agent Skill Workflow

The clipping tool is packaged as an agent skill so users can invoke the engine
with `/clip` instead of manually running every CLI stage.

## Commands

- `/clip <video>` runs the full local workflow for a video path, attached video
  file, direct video URL, or platform URL when `yt-dlp` is installed.
- `/clip-config` checks or writes local defaults such as output directory,
  ratios, candidate count, approval threshold, and Hugging Face token status.

Command shims live in `commands/`. The skill contract lives in root `SKILL.md`.

## Runtime Split

The skill keeps expensive model usage inside the hosting harness:

- Local engine: probe, transcription, diarization, vision analysis, candidate
  mining, review manifest generation, crop planning, caption planning, FFmpeg
  rendering.
- Harness model: semantic scoring of `scoring-request.json` into
  `scoring-response.json`.
- Skill helper: config loading, source preparation, approval selection, and
  final output reporting.

## Helper Script

`scripts/clip_skill.py` provides deterministic operations that agents should
not reimplement. Resolve the skill root against `CLAUDE_PLUGIN_ROOT` when
installed as a plugin, otherwise the repo checkout:

```bash
CLIPPER_ROOT="${CLIPPER_ROOT:-${CLAUDE_PLUGIN_ROOT:-$PWD}}"
CLIPPER_PYTHON="${CLIPPER_PYTHON:-$CLIPPER_ROOT/.venv/bin/python}"
[ -x "$CLIPPER_PYTHON" ] || CLIPPER_PYTHON="$(command -v python3)"
"$CLIPPER_PYTHON" "$CLIPPER_ROOT/scripts/clip_skill.py" config-check
"$CLIPPER_PYTHON" "$CLIPPER_ROOT/scripts/clip_skill.py" config-write --output-dir ~/Documents/ClipperTool
"$CLIPPER_PYTHON" "$CLIPPER_ROOT/scripts/clip_skill.py" prepare "$SOURCE" --ratios "9:16,1:1"
"$CLIPPER_PYTHON" "$CLIPPER_ROOT/scripts/clip_skill.py" approve "$REVIEW_MANIFEST" --top 3 --min-score 0.70
"$CLIPPER_PYTHON" "$CLIPPER_ROOT/scripts/clip_skill.py" outputs "$RENDER_REPORT"
```

The helper reads `~/.config/clipper-tool/.env`, then lets real environment
variables override file values. URL sources are validated with `ffprobe` at
download time so an HTML error page saved as `.mp4` fails at the boundary
instead of crashing the mine stage.

## Config Keys

- `CLIPPER_OUTPUT_DIR`: default output root.
- `CLIPPER_RATIOS`: comma-separated subset of `9:16`, `1:1`, `16:9`; defaults
  to all.
- `CLIPPER_MAX_CANDIDATES`: candidate count requested from mining.
- `CLIPPER_APPROVE_TOP`: max clips approved by the skill helper.
- `CLIPPER_MIN_SCORE`: approval threshold; if no clip clears it, the best clip
  is approved so the user gets output.
- `HF_TOKEN`: Hugging Face token for WhisperX / pyannote diarization.

## End-To-End Shape

1. `/clip` parses source and options.
2. `prepare` writes a job JSON.
3. `clipper.cli run --stage mine` writes `scoring-request.json`.
4. The harness model writes `scoring-response.json`.
5. `clipper.cli run --stage review` writes `review-manifest.json`.
6. `approve` marks selected candidates with `approved: true`.
7. `clipper.cli run --stage render` writes final MP4s and `render-report.json`.
8. `outputs` formats the rendered paths for the final response.

## Output Location

Final MP4s are under:

```text
<output_dir>/jobs/<job_id>/renders/<clip_id>/
```

`render-report.json` is the index for all rendered clips and ratios.
