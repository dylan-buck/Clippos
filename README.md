# Clipper Tool

Local-first clipping engine and agent skill for Claude Code, Codex, and Hermes
Agent.

## Local Setup

Requirements:

- Python 3.12
- FFmpeg and `ffprobe` on your `PATH`

Install the project and dev tools:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

To run the real transcription + diarization pipeline, also install the engine extras:

```bash
pip install -e ".[engine,dev]"
```

The engine extras pull in:

- `whisperx` — large-v3 ASR + wav2vec2 forced alignment
- `pyannote.audio` — `speaker-diarization-3.1`
- `opencv-python` — frame sampling and Farnebäck optical flow for motion scoring
- `scenedetect` — `ContentDetector` for shot-change detection
- `mediapipe` — face detection for framing anchors

Diarization requires a Hugging Face account:

1. Create a token at <https://huggingface.co/settings/tokens>.
2. Accept the license for `pyannote/speaker-diarization-3.1` at
   <https://hf.co/pyannote/speaker-diarization-3.1>.
3. Export the token before running a job:

   ```bash
   export HF_TOKEN=hf_...
   ```

   `HUGGING_FACE_HUB_TOKEN` and `HUGGINGFACE_HUB_TOKEN` are also accepted.

Run the checks used in local development:

```bash
ruff check .
.venv/bin/pytest -v
```

Run the gated real-video E2E check when you want to validate the production
path on an actual file:

```bash
pip install -e ".[engine,dev]"
export HF_TOKEN=hf_...
export CLIPPER_E2E_VIDEO=/absolute/path/to/5-10-minute-video.mp4
.venv/bin/pytest -m e2e -v
```

That test runs real probe, WhisperX/pyannote transcription, OpenCV/MediaPipe
vision analysis, JSON scoring handoff, approval, and FFmpeg rendering. It skips
cleanly unless `CLIPPER_E2E_VIDEO`, FFmpeg/ffprobe, engine dependencies, and a
Hugging Face token are available.

## CLI Flow

The CLI currently exposes two commands:

- `python -m clipper.cli version`
- `python -m clipper.cli run /absolute/path/job.json [--stage mine|review|render|auto]`

`run` reads a job file, validates it against the shared `ClipperJob` contract, and executes the requested pipeline stage. `--stage` defaults to `auto`.

Minimal job file:

```json
{
  "video_path": "/absolute/path/input.mp4",
  "output_dir": "/absolute/path/output"
}
```

### Stages

- `mine` — ingest + transcribe + vision + mining, then writes
  `scoring-request.json` and prints its path.
- `review` — consumes an existing `scoring-request.json` plus a matching
  `scoring-response.json` (or cached scores) and writes `review-manifest.json`.
- `render` — consumes `review-manifest.json`, builds per-clip `RenderManifest`
  plans for candidates marked `"approved": true`, and shells out to FFmpeg to
  produce 9:16 / 1:1 / 16:9 MP4s with ASS caption sidecars. Emits
  `render-report.json`; exits with an error when no candidates are approved.
- `auto` — runs `mine`, and if the harness has already written the response
  file, continues into `review` in the same invocation. `auto` does not chain
  into render; the render stage must be invoked explicitly after approval.

All workspace artifacts land under `/absolute/path/output/jobs/<job_id>/`:

```text
jobs/<job_id>/scoring-request.json
jobs/<job_id>/scoring-response.json         # produced by the harness
jobs/<job_id>/scoring-cache/<hash>.json     # per-clip cache
jobs/<job_id>/review-manifest.json
jobs/<job_id>/renders/<clip_id>/render-manifest.json
jobs/<job_id>/renders/<clip_id>/<clip_id>-9x16.mp4
jobs/<job_id>/renders/<clip_id>/<clip_id>-1x1.mp4
jobs/<job_id>/renders/<clip_id>/<clip_id>-16x9.mp4
jobs/<job_id>/renders/<clip_id>/<clip_id>-*.ass
jobs/<job_id>/render-report.json
```

### Harness workflow

Scoring is delegated to the surrounding agent harness; the clipper never calls
an LLM API directly and does not use any provider SDK or API key.

1. Run `python -m clipper.cli run job.json --stage mine` to emit
   `scoring-request.json`. The request embeds the rubric prompt, versioned
   rubric id, and the strict JSON schema the response must satisfy.
2. The harness (Claude Code, Codex, or Hermes Agent) loads the request via its
   wrapper helper (e.g. `claude_load_scoring_request`), scores every clip with
   the in-session model, and writes `scoring-response.json` back through
   `claude_write_scoring_response` (or the `codex_*`/`hermes_*` equivalents).
3. Run `python -m clipper.cli run job.json --stage review` (or rerun `auto`)
   to merge scores into `review-manifest.json`.
4. Review `review-manifest.json` and set `"approved": true` on the candidates
   to export.
5. Run `python -m clipper.cli run job.json --stage render` to produce final
   MP4s for approved candidates only.

See [docs/architecture/scoring-handoff.md](docs/architecture/scoring-handoff.md) for the full rubric, schema, and caching rules.

## Agent Skill Flow

The repo can also be used as an agent skill. The skill entrypoint is
[SKILL.md](SKILL.md), with slash-command shims in [commands/clip.md](commands/clip.md)
and [commands/clip-config.md](commands/clip-config.md).

Target invocation shape:

```text
/clip /absolute/path/video.mp4
/clip https://example.com/video.mp4 --ratios 9:16,1:1 --clips 2
/clip-config --output-dir ~/Documents/ClipperTool --ratios 9:16,1:1,16:9
```

For attached videos, the agent should resolve the attachment to a local file
path and pass that path into `/clip`. Direct video URLs are downloaded with the
skill helper. Platform URLs can work when `yt-dlp` is installed.

The helper script resolves `CLIPPER_ROOT` against `CLAUDE_PLUGIN_ROOT` when the
skill is installed as a plugin, and falls back to the repo checkout when run
locally:

```bash
CLIPPER_ROOT="${CLIPPER_ROOT:-${CLAUDE_PLUGIN_ROOT:-$PWD}}"
CLIPPER_PYTHON="${CLIPPER_PYTHON:-$CLIPPER_ROOT/.venv/bin/python}"
[ -x "$CLIPPER_PYTHON" ] || CLIPPER_PYTHON="$(command -v python3)"
"$CLIPPER_PYTHON" "$CLIPPER_ROOT/scripts/clip_skill.py" config-check
"$CLIPPER_PYTHON" "$CLIPPER_ROOT/scripts/clip_skill.py" prepare "$SOURCE"
"$CLIPPER_PYTHON" "$CLIPPER_ROOT/scripts/clip_skill.py" approve "$REVIEW_MANIFEST" --top 3 --min-score 0.70
"$CLIPPER_PYTHON" "$CLIPPER_ROOT/scripts/clip_skill.py" outputs "$RENDER_REPORT"
```

Skill configuration is stored at `~/.config/clipper-tool/.env`. Supported keys:

```env
CLIPPER_OUTPUT_DIR=~/Documents/ClipperTool
CLIPPER_RATIOS=9:16,1:1,16:9
CLIPPER_MAX_CANDIDATES=12
CLIPPER_APPROVE_TOP=3
CLIPPER_MIN_SCORE=0.70
HF_TOKEN=hf_...
```

The skill renders all three ratios by default because rendering is deterministic
and does not use the harness model. If the user asks for fewer formats, the job
uses `output_profile.ratios` and the render stage only writes those requested
outputs.

## Current v1 Limitations

- Transcription + diarization (WhisperX + pyannote) are wired and cached at
  `<workspace>/transcript.json`, keyed by model name.
- Vision analysis (OpenCV frame sampling, PySceneDetect shot changes, MediaPipe
  face detection, Farnebäck optical flow, OneEuro trajectory smoothing) is wired
  and cached at `<workspace>/vision.json`, keyed by adapter model.
- Scoring runs only when the surrounding harness writes a valid
  `scoring-response.json`; the clipper itself does not invoke any LLM.
- The render stage requires `ffmpeg` on your `PATH` (libx264 + AAC). The render
  manifest carries a per-clip `mode` (`TRACK` or `GENERAL`) derived from the
  vision timeline. `TRACK` builds a virtual-camera crop that holds inside a
  safe zone and pans at a bounded rate (piecewise-linear ffmpeg crop
  expression, snap on shot change). `GENERAL` runs a blurred-background
  composition for clips with no clear single subject.
- Caption styling is preset-driven via `output_profile.caption_preset`
  (`hook-default` / `bottom-creator` / `bottom-compact` / `lower-third-clean`
  / `center-punch` / `top-clean`). Each preset resolves to a `CaptionStyle`
  (alignment, bold, font/margin ratios, outline, shadow) in
  `pipeline/caption_styles.py` and drives the ASS Style line emitted by the
  renderer.
- `auto` does not run the render stage; invoke `--stage render` explicitly to
  produce final MP4s after marking candidates approved in `review-manifest.json`.
- Packaging (titles, thumbnail overlay lines, social caption, hashtags, hooks)
  is a separate post-render step driven by `/clip-package` — it reuses the
  scoring-style request/response handoff so the harness model authors the pack
  and the clipper validates + persists per-clip `package.json` artifacts next
  to each rendered MP4. See
  [docs/architecture/package-handoff.md](docs/architecture/package-handoff.md).
