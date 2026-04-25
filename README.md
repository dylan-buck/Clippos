# Clipper Tool

Turn any long-form video into captioned, viral-ready social clips with a
single `/clip` call in your agent. Ships as a skill for
[Hermes](https://hermes-agent.nousresearch.com), Claude Code, and Codex — and
runs in any harness that can execute a Python script and read JSON.

The engine does the hard media work locally (transcription, diarization, face
detection, optical-flow motion scoring, virtual-camera cropping, ASS caption
burn-in, multi-ratio render). The active agent's model handles the
*judgement* work (which clips are worth posting, titles, captions, hashtags)
via a JSON handoff — so the clipper never locks you into a specific
provider, inherits your agent's memory + preferences, and learns from your
keep/skip decisions over time.

Designed Hermes-first. Works anywhere.

## Install

| Harness            | Install                                                                             | Command surface                          |
| ------------------ | ----------------------------------------------------------------------------------- | ---------------------------------------- |
| **Hermes**         | `ln -s /absolute/path/to/clipping-tool ~/.hermes/skills/clip`                       | `/clip`, `/clip config`, `/clip package` |
| **Claude Code**    | `/plugin install ./.claude-plugin` (or copy the repo into `~/.claude/skills/clip`)  | `/clip`, `/clip-config`, `/clip-package` |
| **Codex**          | Load `.codex-plugin/plugin.json` via your Codex plugin loader                       | `/clip`, `/clip-config`, `/clip-package` |
| **Any harness**    | Clone the repo, export `CLIPPER_ROOT=/abs/path/to/clipping-tool`, run the scripts   | `hermes_clip.py advance --source ...`    |

All four install paths resolve to the same `SKILL.md` and the same helper
scripts. After install, see the per-harness steps below, then run the
[local dev setup](#local-dev-setup) once to get `.venv` + `ffmpeg`.

### Hermes

```bash
# From the project root
mkdir -p ~/.hermes/skills
ln -s "$(pwd)" ~/.hermes/skills/clip
```

Start a fresh Hermes session and the `/clip` skill appears automatically.
Hermes reads `SKILL.md` directly and substitutes `${HERMES_SKILL_DIR}` with
the installed skill directory. Typical flow:

```text
/clip /absolute/path/video.mp4
/clip https://cdn.discordapp.com/attachments/... --ratios 9:16,1:1 --clips 2
/clip config --output-dir ~/Documents/ClipperTool --hf-token hf_...
/clip package
```

Attachment URLs dropped into Discord/Telegram are detected and downloaded
directly (yt-dlp is skipped for signed CDN URLs).

### Claude Code

If you have Claude Code's plugin system:

```bash
# Quickest — point Claude Code at the local plugin
/plugin install /absolute/path/to/clipping-tool/.claude-plugin
```

Or copy the repo into `~/.claude/skills/clip`. Claude Code exposes three
slash commands via the `commands/*.md` shims:

```text
/clip /absolute/path/video.mp4
/clip-config --output-dir ~/Documents/ClipperTool --hf-token hf_...
/clip-package
```

Claude Code substitutes `${CLAUDE_PLUGIN_ROOT}` in the prologue
automatically.

### Codex

Codex follows the same plugin shape as Claude Code. Point your Codex plugin
loader at `.codex-plugin/plugin.json` (or symlink the repo into Codex's
skill directory). Commands are identical to Claude Code: `/clip`,
`/clip-config`, `/clip-package`.

### Any other harness (generic)

If you're running a harness without a built-in plugin system (custom agent
framework, bare terminal, a provider SDK), install manually:

```bash
git clone https://github.com/dylan-buck/clipping-tool
cd clipping-tool
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[engine,dev]"
export CLIPPER_ROOT="$(pwd)"
```

Then drive the pipeline with `hermes_clip.py`:

```bash
"$CLIPPER_ROOT/.venv/bin/python" "$CLIPPER_ROOT/scripts/hermes_clip.py" \
  advance --source /absolute/path/video.mp4
```

The script prints structured JSON with a `next_action`: `score`, `package`,
`done-renders`, `done-package`, `error`, or `configure`. Your harness reads
the JSON, writes the scoring/packaging response when prompted, then calls
`advance --workspace "$WORKSPACE"` again to continue.

## What it does

One concrete example. You have a 45-minute podcast recording. In your agent:

```text
/clip ~/Downloads/podcast.mp3.mp4 --ratios 9:16 --clips 3
```

The skill:

1. Transcribes + diarizes locally (WhisperX + pyannote).
2. Analyzes vision: scene cuts, face positions, optical-flow motion.
3. Mines 12 candidate 20–60 second windows with strong hooks, payoffs, and
   spike signals (controversy, emotional beat, unusually useful claim, etc.).
4. Asks your agent's active model to score each candidate against a fixed
   rubric (hook, shareability, standalone clarity, payoff, delivery energy,
   quotability) plus creator-profile cues from past runs.
5. Auto-approves the top 3 above your quality threshold (or falls back to
   the best).
6. Virtual-camera-crops each approved clip to 9:16, burns ASS captions in
   the configured preset, renders an H.264/AAC mp4.
7. Returns the workspace path and mp4 paths to your agent.

Optionally follow up with `/clip package` to generate title candidates,
thumbnail overlay lines, social captions, hashtags, and opening-line hooks
for every rendered clip.

## What makes it unique

- **Harness-agnostic.** The clipper never calls an LLM directly — it hands
  every semantic decision to whatever model your agent is running. Same
  engine, any provider.
- **Chat-native.** Drop a video in Hermes Discord or Telegram, get back
  finished mp4s in the same thread. Discord CDN and Telegram bot-file URLs
  are detected automatically.
- **Self-improving creator profile.** After each run, record which clips you
  posted vs. skipped (`/clip feedback` or programmatically via
  `hermes_clip.py feedback`). The skill aggregates patterns across runs
  (length bias, spike-category preference, ratio preference, score
  disagreement) with confidence tiers and surfaces them to the next scoring
  handoff. Rules can be promoted into the harness's memory.
- **Local-first, zero-config.** Transcription, diarization, vision, and
  rendering all run on your machine with no API keys, no HuggingFace
  token, and no license click-throughs. Default speaker diarization uses
  silero-VAD + SpeechBrain ECAPA-TDNN (Apache 2.0 / CC-BY-4.0, public
  weights). The pyannote 3.1 upgrade stays available as an opt-in for
  users who already have an HF token.
- **Deterministic engine, judgement delegated.** The clipper validates every
  handoff against a strict JSON schema with `clip_id`/`clip_hash` integrity
  checks, so model outputs can't silently corrupt a run.

## How it works

The pipeline is a three-stage state machine: `mine` → `review` → `render`.
Each stage writes a JSON artifact in the workspace. Scoring and packaging
are model handoffs: the engine writes a request JSON, the agent writes a
response JSON, the engine validates and continues.

```text
/clip video.mp4
    │
    ├─→ mine       → scoring-request.json
    │               (agent scores → scoring-response.json)
    ├─→ review     → review-manifest.json (auto-approves top N)
    ├─→ render     → render-report.json + renders/<clip>/<clip>-{9x16,1x1,16x9}.mp4
    │
    └─ /clip package
        ├─→ package-prompt → package-request.json
        │                    (agent packages → package-response.json)
        └─→ package-save    → renders/<clip>/package.json
                              + package-report.json
```

The full skill flow, including creator-profile memory, feedback loop, and
raw-primitive fallback, is in [SKILL.md](SKILL.md).

## Local dev setup

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
- `opencv-python` — frame sampling and colour-space conversion
- `scenedetect` — `ContentDetector` for shot-change detection
- `retina-face` + `tf-keras` — RetinaFace-ResNet50 for framing anchors
  (MIT; pulls TensorFlow as a dep — model weights are ~119 MB on first run)
- `torch` + `torchvision` — RAFT optical flow for motion scoring
  (auto-selects `mps` / `cuda` / `cpu`)
- `speechbrain` + `silero-vad` + `scikit-learn` — open-source speaker
  diarization (silero-VAD → ECAPA-TDNN embeddings → spectral clustering).
  Public weights, no token, no license click-through.

### Diarization (zero-config)

Diarization runs out of the box. The default stack uses silero-VAD plus
SpeechBrain's `speechbrain/spkrec-ecapa-voxceleb` ECAPA-TDNN model — both
are public, both auto-cache locally on first use, neither requires a
HuggingFace token.

If you want the higher-quality `pyannote/speaker-diarization-3.1` upgrade,
opt in with `CLIPPER_DIARIZER=pyannote`. That path requires a one-time
HuggingFace setup:

1. Create a token at <https://huggingface.co/settings/tokens>.
2. Accept the license for `pyannote/speaker-diarization-3.1` at
   <https://hf.co/pyannote/speaker-diarization-3.1>.
3. Export the token before running a job:

   ```bash
   export HF_TOKEN=hf_...
   export CLIPPER_DIARIZER=pyannote
   ```

   `HUGGING_FACE_HUB_TOKEN` and `HUGGINGFACE_HUB_TOKEN` are also accepted
   for the token. `CLIPPER_DIARIZER=off` skips diarization entirely.

Run the checks used in local development:

```bash
ruff check .
.venv/bin/pytest -v
```

Run the gated real-video E2E check when you want to validate the production
path on an actual file:

```bash
pip install -e ".[engine,dev]"
export CLIPPER_E2E_VIDEO=/absolute/path/to/5-10-minute-video.mp4
.venv/bin/pytest -m e2e -v
```

That test runs real probe, WhisperX transcription, the open-source
diarizer, RetinaFace-ResNet50 face detection, RAFT optical flow, JSON
scoring handoff, approval, and FFmpeg rendering. It skips cleanly unless
`CLIPPER_E2E_VIDEO`, FFmpeg/ffprobe, and engine dependencies are
available. To exercise the pyannote path instead, also set `HF_TOKEN`
plus `CLIPPER_DIARIZER=pyannote`.

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

## Configuration

Skill configuration lives at `~/.config/clipper-tool/.env`. Write it
through the skill rather than hand-editing:

```bash
"$CLIPPER_PYTHON" "$CLIPPER_ROOT/scripts/clip_skill.py" config-write \
  --output-dir "$HOME/Documents/ClipperTool" \
  --ratios "9:16,1:1,16:9" \
  --approve-top 3 \
  --min-score 0.70
```

Supported keys (all optional):

```env
CLIPPER_OUTPUT_DIR=~/Documents/ClipperTool
CLIPPER_RATIOS=9:16,1:1,16:9
CLIPPER_MAX_CANDIDATES=12
CLIPPER_APPROVE_TOP=3
CLIPPER_MIN_SCORE=0.70
# Optional. Default diarizer is the open-source SpeechBrain stack (no token).
# Set CLIPPER_DIARIZER=pyannote and HF_TOKEN to opt into the pyannote upgrade.
CLIPPER_DIARIZER=speechbrain
HF_TOKEN=hf_...
```

The skill renders all three ratios by default because rendering is
deterministic and does not use the agent's model. Narrow the set with
`--ratios` only when the user explicitly asks.

## Demo (two-minute flow)

Pick any known-good local video 5–10 minutes long.

1. **Install.** `ln -s $(pwd) ~/.hermes/skills/clip` (or the Claude
   Code / Codex equivalent above).
2. **Configure.** In your agent, run `/clip config --output-dir
   ~/Documents/ClipperTool` (Hermes) or `/clip-config ...` (Claude Code /
   Codex). Writes the `.env`. **No HuggingFace token needed** — diarization
   uses the open-source SpeechBrain stack by default.
3. **Clip.** Run `/clip ~/Downloads/sample-talk.mp4 --ratios 9:16,1:1
   --clips 2`. The agent scores each candidate with its active model,
   the skill auto-approves the top 2 and renders them, and the agent
   reports back the workspace + mp4 paths.
4. **Package.** Run `/clip package`. Produces per-clip `package.json`
   with titles, thumbnail overlay lines, social caption, hashtags, and
   opening-line hooks.
5. **Learn.** Tell the agent which clips you actually posted:
   `hermes_clip.py feedback <workspace> --kept c1 --skipped c2 --note
   c2='too long'`. The next `/clip` run will surface patterns in the
   scoring handoff.

## Current v1 Limitations

- Transcription + diarization (WhisperX + pyannote) are wired and cached at
  `<workspace>/transcript.json`, keyed by model name.
- Vision analysis (OpenCV frame sampling, PySceneDetect shot changes,
  RetinaFace-ResNet50 face detection via `retina-face`, torchvision RAFT
  optical flow, OneEuro trajectory smoothing) is wired and cached at
  `<workspace>/vision.json`, keyed by adapter model. RAFT auto-selects the
  best available PyTorch device (`mps` / `cuda` / `cpu`).
- Scoring runs only when the surrounding harness writes a valid
  `scoring-response.json`; the clipper itself does not invoke any LLM.
- The render stage requires `ffmpeg` on your `PATH` (libx264 + AAC + libass /
  `ass` filter for caption burn-in). The render
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
