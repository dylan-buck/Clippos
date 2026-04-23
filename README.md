# Clipper Tool

Local-first clipping engine for Claude Code, Codex, and Hermes Agent.

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
  plans, and shells out to FFmpeg to produce 9:16 / 1:1 / 16:9 MP4s with ASS
  caption sidecars. Emits `render-report.json`.
- `auto` — runs `mine`, and if the harness has already written the response
  file, continues into `review` in the same invocation. `auto` does not chain
  into render; the render stage must be invoked explicitly (M1.6 will add a
  human approval gate).

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

See [docs/architecture/scoring-handoff.md](docs/architecture/scoring-handoff.md) for the full rubric, schema, and caching rules.

## Current v1 Limitations

- Transcription + diarization (WhisperX + pyannote) are wired and cached at
  `<workspace>/transcript.json`, keyed by model name.
- Vision analysis (OpenCV frame sampling, PySceneDetect shot changes, MediaPipe
  face detection, Farnebäck optical flow, OneEuro trajectory smoothing) is wired
  and cached at `<workspace>/vision.json`, keyed by adapter model.
- Scoring runs only when the surrounding harness writes a valid
  `scoring-response.json`; the clipper itself does not invoke any LLM.
- The render stage requires `ffmpeg` on your `PATH` (libx264 + AAC). Crops are
  static per clip — driven by OneEuro-smoothed face anchors — and captions are
  rendered via the ASS subtitle filter.
- `auto` does not run the render stage; invoke `--stage render` explicitly to
  produce final MP4s (the M1.6 approval loop will gate this automatically).
