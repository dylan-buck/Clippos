---
name: clip
description: Local video clipping automation for /clip requests, attached video files, video links, social clip generation, captioned shorts, crop/framing, and rendered 9:16, 1:1, or 16:9 exports. Use this skill whenever the user wants an agent to turn a video into high-potential clips, score clips with the current harness model, approve selected clips, and render final MP4 outputs.
---

# Clip Skill

Use this skill to run the local clipping engine end-to-end from a video path,
attached video file, or video link. The engine does deterministic media work
locally; the current agent supplies the semantic scoring step by reading
`scoring-request.json` and writing `scoring-response.json`.

## Inputs

Accept:

- A local video path, including a path resolved from an attached video file.
- A direct `http` / `https` video URL.
- A platform video URL when `yt-dlp` is installed.

Optional user intent:

- Ratios: default all three (`9:16`, `1:1`, `16:9`). Respect explicit requests
  such as "vertical only", "square", "wide", or `--ratios 9:16,1:1`.
- Clip count: default to config `CLIPPER_APPROVE_TOP` or 3.
- Quality threshold: default to config `CLIPPER_MIN_SCORE` or 0.70.
- Output directory: default to config `CLIPPER_OUTPUT_DIR` or
  `~/Documents/ClipperTool`.

If no video source is present, ask one short question for the video link or
file path.

## Preflight

Run this before the first clip job in a session:

```bash
CLIPPER_PYTHON="${CLIPPER_PYTHON:-$PWD/.venv/bin/python}"
if [ ! -x "$CLIPPER_PYTHON" ]; then CLIPPER_PYTHON="$(command -v python3)"; fi
"$CLIPPER_PYTHON" scripts/clip_skill.py config-check
```

If `HF_TOKEN` is missing and no cached transcript exists, use `/clip-config` or
ask the user for setup. The real pipeline needs FFmpeg, ffprobe, engine extras,
and a Hugging Face token for WhisperX/pyannote diarization.

## Main Workflow

1. Prepare a job from the source:

```bash
"$CLIPPER_PYTHON" scripts/clip_skill.py prepare "$SOURCE" --ratios "9:16,1:1,16:9"
```

Use narrower ratios only when the user asks. The command prints JSON containing
`job_path`, `approve_top`, and `min_score`.

2. Mine candidates:

```bash
"$CLIPPER_PYTHON" -m clipper.cli run "$JOB_PATH" --stage mine
```

This writes `scoring-request.json` under the job workspace.

3. Score every clip as the harness model:

- Read `scoring-request.json`.
- Follow its embedded `rubric_prompt` and `response_schema`.
- Score every `ClipBrief`; do not invent `clip_id` or `clip_hash`.
- Use transcript, mining signals, spike categories, penalties, and standalone
  clarity evidence.
- Write a valid `scoring-response.json` beside the request.

4. Build the review manifest:

```bash
"$CLIPPER_PYTHON" -m clipper.cli run "$JOB_PATH" --stage review
```

5. Approve selected clips:

```bash
"$CLIPPER_PYTHON" scripts/clip_skill.py approve "$REVIEW_MANIFEST" --top "$APPROVE_TOP" --min-score "$MIN_SCORE"
```

The helper approves top scoring clips above threshold. If none clear the
threshold, it approves the best clip so the user gets a concrete output.

6. Render approved clips:

```bash
"$CLIPPER_PYTHON" -m clipper.cli run "$JOB_PATH" --stage render
```

Rendering is deterministic and does not spend model tokens. Render all three
ratios by default; render only requested ratios when the user explicitly asks.

7. Report outputs:

```bash
"$CLIPPER_PYTHON" scripts/clip_skill.py outputs "$RENDER_REPORT"
```

Return the final MP4 paths grouped by clip and ratio. Include the job workspace
path and mention if any requested ratio was not rendered.

## Config Workflow

Use `/clip-config` when setup is missing or the user wants defaults changed.
The helper stores config at `~/.config/clipper-tool/.env`:

```bash
"$CLIPPER_PYTHON" scripts/clip_skill.py config-write \
  --output-dir "$HOME/Documents/ClipperTool" \
  --ratios "9:16,1:1,16:9" \
  --max-candidates 12 \
  --approve-top 3 \
  --min-score 0.70
```

If the user gives a Hugging Face token, add `--hf-token "$HF_TOKEN"`. Treat the
token as sensitive and do not print it back.

## Output Contract

End the response with:

- Workspace path.
- Approved clip IDs and titles when available.
- Rendered MP4 paths for each ratio.
- Any setup or render failures with the exact command that failed.

Do not call the job production-ready until a real-video `/clip` run has
completed on the user's machine with engine extras and `HF_TOKEN` configured.
