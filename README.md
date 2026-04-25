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

> **Heads up:** first `/clip` run downloads ~3.5 GB of model weights and
> the pipeline is compute-heavy. Read [Hardware requirements](#hardware-requirements)
> before installing — 16 GB RAM is the practical floor.

Shortest path:

```bash
curl -fsSL https://raw.githubusercontent.com/dylan-buck/clipping-tool/main/install.sh | bash
```

That clones the repo into `~/.local/share/clipping-tool`, creates `.venv`,
installs the engine extras, and links the `clip` skill into Hermes, Claude
Code, and Codex skill directories. Tune it with environment variables, e.g.
`CLIPPER_HARNESS=hermes`, `CLIPPER_INSTALL_DIR=/path/to/clipping-tool`, or
`CLIPPER_EXTRAS=none`.

| Harness            | Install                                                                             | Command surface                          |
| ------------------ | ----------------------------------------------------------------------------------- | ---------------------------------------- |
| **Hermes**         | `CLIPPER_HARNESS=hermes curl -fsSL .../install.sh \| bash` or symlink manually      | `/clip`, `/clip config`, `/clip package` |
| **Claude Code**    | `CLIPPER_HARNESS=claude curl -fsSL .../install.sh \| bash` or plugin install        | `/clip`, `/clip-config`, `/clip-package` |
| **Codex**          | `CLIPPER_HARNESS=codex curl -fsSL .../install.sh \| bash` or plugin loader          | `/clip`, `/clip-config`, `/clip-package` |
| **Any harness**    | Clone the repo, export `CLIPPER_ROOT=/abs/path/to/clipping-tool`, run the scripts   | `hermes_clip.py advance --source ...`    |

All four install paths resolve to the same `SKILL.md` and the same helper
scripts. The one-liner does the local engine setup; manual installs can use
the per-harness steps below and then the [local dev setup](#local-dev-setup).

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
/clip config --output-dir ~/Documents/ClipperTool
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
/clip-config --output-dir ~/Documents/ClipperTool
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

The script prints structured JSON with a `next_action`: `brief`, `score`,
`package`, `done-renders`, `done-package`, `error`, or `configure`. Your
harness reads the JSON, writes the requested response file when prompted,
then calls `advance --workspace "$WORKSPACE"` again to continue.

## Hardware requirements

The pipeline runs Whisper large-v3 transcription, RetinaFace face
detection, and RAFT optical flow locally — accurate but compute-heavy.
Calibrate before installing.

**Minimum (works, may be slow):**

- macOS Apple Silicon (M1+) with 16 GB unified memory, OR
- Linux x86_64 with 16 GB RAM (NVIDIA GPU strongly recommended)
- 10 GB free disk: ~3.5 GB model weights + ~2 GB vendored ffmpeg +
  workspace headroom

**Recommended:**

- Apple Silicon M2 Pro / M3 / M4 with 32 GB, OR NVIDIA RTX 30-series+
- 50 GB free disk if you plan to keep multiple workspaces

**Expected runtime on a 10-minute source video, M2 Pro 32 GB:**

- First run only: ~5 min model downloads (Whisper, SpeechBrain ECAPA,
  RetinaFace, RAFT — cached after that)
- Mining (transcribe + diarize + vision): 3–5 min
- Render: 1–2 min per ratio (so 3–6 min for the default 9:16 + 1:1 + 16:9)
- **Your fan will spin up.** Vision (RAFT optical flow on every sampled
  frame pair) is the loudest stage. This is normal.

**Scaling with duration:**

- 30-min video: ~10–15 min mining, ~3–6 min render per ratio
- 60-min video: ~25–40 min mining; 16 GB Macs may hit memory pressure
  during transcription — close Chrome and Slack first

**CPU-only machines (no GPU, no MPS):**

Everything still works, but expect 3–10× slower. A 10-min video may
take 30–45 min total.

Source videos are auto-capped to 1080p before transcription, so 4K @ 60 fps
inputs do not blow up memory — only duration scales peak RAM.

## Demo (5-minute flow)

Pick any known-good local video 5–10 minutes long.

1. **Install.** `curl -fsSL https://raw.githubusercontent.com/dylan-buck/clipping-tool/main/install.sh | bash`.
2. **Configure** (optional). In your agent, run `/clip config --output-dir
   ~/Documents/ClipperTool` (Hermes) or `/clip-config ...` (Claude Code /
   Codex). Writes the `.env`. **No HuggingFace token needed** —
   diarization uses the open-source SpeechBrain stack by default.
3. **Clip.** Run `/clip ~/Downloads/sample-talk.mp4 --ratios 9:16,1:1`.
   The skill mines candidates locally, the agent first authors a video
   brief from the transcript (one model handoff), then scores each
   candidate, the skill auto-approves the top 5 + renders, and the
   agent reports back the workspace, clips directory, and MP4 paths.
4. **Package.** Run `/clip package`. Produces per-clip `package.json`
   with titles, thumbnail overlay lines, social caption, hashtags, and
   opening-line hooks.
5. **Learn.** Tell the agent which clips you actually posted:
   `hermes_clip.py feedback <workspace> --kept c1 --skipped c2 --note
   c2='too long'`. The next `/clip` run will surface patterns in the
   scoring handoff.

## Configuration

Skill configuration lives at `~/.config/clipper-tool/.env`. Write it
through the skill rather than hand-editing:

```bash
"$CLIPPER_PYTHON" "$CLIPPER_ROOT/scripts/clip_skill.py" config-write \
  --output-dir "$HOME/Documents/ClipperTool" \
  --ratios "9:16,1:1,16:9" \
  --approve-top 5 \
  --min-score 0.70
```

Supported keys (all optional):

```env
CLIPPER_OUTPUT_DIR=~/Documents/ClipperTool   # where MP4s land
CLIPPER_RATIOS=9:16,1:1,16:9                 # default render set
CLIPPER_MAX_CANDIDATES=12                    # mining cap per video
CLIPPER_APPROVE_TOP=5                        # auto-approve top N scores
CLIPPER_MIN_SCORE=0.70                       # threshold for top-N selection

# Optional. Default diarizer is open-source SpeechBrain (no token needed).
# Set CLIPPER_DIARIZER=pyannote and HF_TOKEN to opt into the pyannote upgrade.
CLIPPER_DIARIZER=speechbrain
HF_TOKEN=hf_...
```

Per-job knobs (passed at invocation, not persisted):

- `--ratios 9:16,1:1` — render only the listed ratios
- `--clips 3` — auto-approve the top N (overrides `CLIPPER_APPROVE_TOP`)
- `--min-score 0.6` — lower the auto-approve threshold for this run
- `--max-candidates 8` — cap mining for this run

The skill renders all three ratios by default because rendering is
deterministic and does not use the agent's model. Narrow the set with
`--ratios` only when the user explicitly asks.

## Output locations

By default, all artifacts land under `~/Documents/ClipperTool/jobs/<job_id>/`.
Override with `--output-dir` at job time or set `CLIPPER_OUTPUT_DIR` in
your config. The `<job_id>` is a SHA-1 of the source video path —
re-running on the same path reuses the same workspace and skips
already-cached stages.

Per-job workspace layout:

```text
~/Documents/ClipperTool/jobs/<job_id>/
├── transcript.json              # WhisperX output (cached)
├── vision.json                  # face / motion / scene-cut signals (cached)
├── brief-request.json           # ← engine writes; harness authors brief
├── brief-response.json          # ← harness writes
├── brief-cache.json             # last good brief, survives reruns
├── scoring-request.json         # ← engine writes; harness scores each clip
├── scoring-response.json        # ← harness writes
├── scoring-cache/<hash>.json    # per-clip score cache, keyed by brief context
├── review-manifest.json         # auto-approved candidates
├── render-report.json           # final summary with output paths
└── renders/<clip_id>/
    ├── <clip_id>-9x16.mp4       # final MP4s for each requested ratio
    ├── <clip_id>-1x1.mp4
    ├── <clip_id>-16x9.mp4
    ├── <clip_id>-*.ass          # ASS subtitle sidecars
    ├── render-manifest.json
    └── package.json             # /clip-package output (titles, hashtags, etc.)
```

The MP4s are what you upload. The JSON files are the workspace's audit
trail — they let you re-run any stage without re-mining and they're how
the harness model picks up where it left off across `/clip` invocations.

## What it does

One concrete example. You have a 45-minute podcast recording. In your agent:

```text
/clip ~/Downloads/podcast.mp3.mp4 --ratios 9:16
```

The skill:

1. Transcribes locally with WhisperX large-v3, then diarizes with the
   zero-config SpeechBrain ECAPA + silero-VAD stack (no HF token).
2. Analyzes vision: scene cuts (PySceneDetect), face positions
   (RetinaFace-ResNet50), optical-flow motion (RAFT).
3. Mines 12 candidate 20–60s windows with strong hooks, payoffs, and
   spike signals (controversy, big-number, expert-endorsement, etc.),
   plus an explicit guarantee that detected multi-speaker / interview
   blocks each get at least one candidate even when their windows score
   below the regular floor.
4. Asks the agent's active model to author a one-paragraph **video brief**
   — theme, expected viral patterns, anti-patterns — from the full
   transcript. One model handoff per video; cached for the rest of the
   workspace's life.
5. Asks the model to score every candidate against a fixed rubric (hook,
   shareability, standalone clarity, payoff, delivery energy,
   quotability) plus the brief-derived bias and creator-profile cues
   from past runs.
6. Auto-approves at least the top 5 candidates when the video has enough
   valid windows, filling below the quality threshold only when needed
   to satisfy the minimum.
7. Virtual-camera-crops each approved clip to 9:16, burns ASS captions in
   the configured preset, renders an H.264/AAC mp4.
8. Returns the workspace path and mp4 paths to your agent.

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
- **Self-improving creator profile.** After each run, record which clips
  you posted vs. skipped (`/clip feedback` or programmatically via
  `hermes_clip.py feedback`). The skill aggregates patterns across runs
  (length bias, spike-category preference, ratio preference, score
  disagreement) with confidence tiers and surfaces them to the next
  scoring handoff. Rules can be promoted into the harness's memory.
- **Local-first, zero-config.** Transcription, diarization, vision, and
  rendering all run on your machine with no API keys, no HuggingFace
  token, and no license click-throughs. Default speaker diarization uses
  silero-VAD + SpeechBrain ECAPA-TDNN (Apache 2.0 / CC-BY-4.0, public
  weights). The pyannote 4.x upgrade stays available as an opt-in for
  users who already have an HF token.
- **Video-brief context.** Before per-clip scoring, the model reads the
  full transcript and authors an opinionated frame: theme, expected
  viral patterns, anti-patterns. Per-clip scoring then sees the global
  shape of the video — not just one clip in isolation. Cached per
  workspace so re-running scoring doesn't re-pay the brief cost.
- **Deterministic engine, judgement delegated.** The clipper validates
  every handoff against a strict JSON schema with `clip_id`/`clip_hash`
  integrity checks (the clip hash now folds in the brief context too,
  so brief edits invalidate the relevant cached scores), so model
  outputs can't silently corrupt a run.

## How it works

The pipeline is a state machine. Each stage writes a JSON artifact in
the workspace; deterministic stages run automatically, model-handoff
stages pause for a response file:

```text
/clip video.mp4
    │
    ├─→ mine        → scoring-request.json + brief-request.json
    │                 (transcribes, diarizes, analyzes vision, mines candidates)
    ├─→ brief       (agent authors → brief-response.json)
    │                 ↳ engine embeds brief into scoring-request.json
    ├─→ score       (agent scores → scoring-response.json)
    ├─→ review      → review-manifest.json (auto-approves top N)
    ├─→ render      → render-report.json + renders/<clip>/<clip>-{9x16,1x1,16x9}.mp4
    │
    └─ /clip package
        ├─→ package-prompt → package-request.json (with brief embedded)
        │                    (agent packages → package-response.json)
        └─→ package-save    → renders/<clip>/package.json
                              + package-report.json
```

The full skill flow, including creator-profile memory, the brief
handoff contract, and feedback loop, is in [SKILL.md](SKILL.md).

## Local dev setup

Requirements:

- Python 3.12 (TensorFlow wheels cap at 3.12; pyproject pins `<3.13`)
- FFmpeg and `ffprobe` on your `PATH`, OR engine extras installed
  (the vendored `static-ffmpeg` is auto-used as a fallback)

Install the project and dev tools:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

To run the real transcription + diarization pipeline, also install the
engine extras:

```bash
pip install -e ".[engine,dev]"
```

Or, with `uv` (catches resolver-strict pin issues that pip silently
ignores — recommended for fresh installs):

```bash
uv sync --extra engine --extra dev
```

Run the checks used in local development:

```bash
ruff check .
.venv/bin/pytest -v
```

Run the gated real-video E2E check when you want to validate the
production path on an actual file:

```bash
export CLIPPER_E2E_VIDEO=/absolute/path/to/5-10-minute-video.mp4
.venv/bin/pytest -m e2e -v
```

## CLI Flow (advanced)

For non-harness use, drive the pipeline directly with the CLI:

- `python -m clipper.cli version`
- `python -m clipper.cli run /absolute/path/job.json [--stage mine|brief|review|render|auto]`

`run` reads a job file, validates it against the shared `ClipperJob`
contract, and executes the requested pipeline stage. `--stage` defaults
to `auto`.

Minimal job file:

```json
{
  "video_path": "/absolute/path/input.mp4",
  "output_dir": "/absolute/path/output"
}
```

### Stages

- `mine` — ingest + transcribe + vision + mining, then writes
  `scoring-request.json` (and `brief-request.json` if `video_brief` is
  enabled in the job's `output_profile`).
- `brief` — re-writes `scoring-request.json` with the resolved video
  brief embedded. Requires `brief-response.json` (or a cached brief).
- `review` — consumes an existing `scoring-request.json` plus a matching
  `scoring-response.json` (or cached scores) and writes
  `review-manifest.json`.
- `render` — consumes `review-manifest.json`, builds per-clip
  `RenderManifest` plans for candidates marked `"approved": true`, and
  shells out to FFmpeg to produce the configured ratios + ASS caption
  sidecars. Emits `render-report.json`; exits with an error when no
  candidates are approved.
- `auto` — runs `mine`, then `brief` (when enabled and the response is
  available), then `review`. **Does not chain into render** — that
  must be invoked explicitly. The Hermes `/clip` flow handles
  approve + render automatically; the raw CLI stops at review by
  design.

See [docs/architecture/scoring-handoff.md](docs/architecture/scoring-handoff.md)
for the full rubric, schema, and caching rules.

## Current v1 limitations

- **`--stage auto` does not chain into render.** The CLI's `auto` stage
  runs mine → brief → score → review and stops; render must be invoked
  explicitly. The Hermes `/clip` flow auto-approves + renders past
  review automatically, but raw-CLI users need an extra step.
- **Auto-approval is the default in the agent flow.** The `/clip` skill
  auto-approves the top N scoring candidates above `min_score`, with
  backfill from below-threshold windows when fewer than N qualify.
  There's no required human-review pause in the agent loop. To gate on
  manual review, drive the raw CLI directly: `--stage review`, edit
  `review-manifest.json` to flip `"approved": true` on the candidates
  you want, then `--stage render`.
- **Linux and Windows install paths are not dogfood-verified.** The
  pin set is resolver-clean on macOS arm64 (verified under both pip
  and `uv sync`) but neither install.sh nor the engine extras have been
  cold-installed on Linux x86_64 or Windows. Both should work — TF
  and torch wheels exist for both — but verification is pending. See
  [docs/pre-ship-fixes.md](docs/pre-ship-fixes.md).
- **NVIDIA / CUDA wheels require manual install on Linux.** install.sh
  pulls the CPU `torch==2.8.0` wheel by default. Linux users with
  NVIDIA GPUs need to install the CUDA-suffixed wheel manually
  (e.g. `pip install torch==2.8.0+cu124 --index-url
  https://download.pytorch.org/whl/cu124`). install.sh does not
  auto-detect CUDA.
- **Mining heuristics are English-tuned.** Both monologue keyword
  buckets (controversy, taboo, etc.) and interview keyword buckets
  ("hands down", "I'm long", etc.) are English-only. WhisperX
  large-v3 transcribes other languages, but the candidate miner will
  surface poor windows for non-English content. The harness model's
  brief + scoring can partially compensate, but the heuristics
  themselves are English-first.
- **Job IDs are path-hashed, single source per workspace.** A workspace
  is identified by SHA-1 of the source video path. Re-running on the
  same path reuses the same workspace (good for resume). If you edit
  the source video at the same path, manually delete the workspace
  under `<output_dir>/jobs/<job_id>/` to force a fresh mine. There's
  no batch mode — invoke `/clip` (or `hermes_clip.py advance --source
  ...`) in a loop for multiple videos.
- **Brief stage adds one model handoff per video.** Disable via
  `output_profile.video_brief: false` in the job for the legacy
  single-handoff flow. The brief is cached per workspace, so the cost
  is paid once per video, not once per scoring run.
