# Hermes Setup Guide for Clippos

This guide covers installing Clippos as a self-contained skill in your
Hermes workspace. After install, the `/clip` command surfaces in any
Hermes session.

Clippos is published to Claude Code and Codex via their native plugin
marketplaces (see [README](README.md)) — Hermes installs work
differently because Hermes is a self-contained workspace and there is
no Hermes plugin marketplace yet.

## Prerequisites

1. **Hermes installed** — see https://github.com/mercurial-tf/hermes
2. **Python 3.12** — `brew install python@3.12` on macOS, or
   `apt install python3.12 python3.12-venv` on Debian/Ubuntu. TensorFlow
   wheels (pulled transitively via `retina-face`) cap at 3.12, so
   3.13/3.14 will not resolve. Our pyproject pins `<3.13` to fail fast.
3. **16 GB RAM minimum** + ~10 GB free disk (~3.5 GB model weights, ~700 MB
   pip wheels, workspace headroom). See [Hardware requirements](README.md#hardware-requirements).

## Installation

Clone the repo into your Hermes skill directory and run the bootstrap
script:

```bash
git clone https://github.com/dylan-buck/Clippos ~/.hermes/skills/clip
bash ~/.hermes/skills/clip/scripts/bootstrap-venv.sh
```

The bootstrap script:

1. Verifies Python 3.12 is available (errors out with brew/apt hints
   otherwise).
2. Creates `~/.hermes/skills/clip/.venv/` and installs the engine
   extras (~5 min, ~700 MB of pip wheels).
3. Persists `CLIPPOS_ROOT` to `~/.config/clippos/.env` so the SKILL
   prologue can resolve the install path on every `/clip` call.

After bootstrap completes, start a fresh Hermes session and the
`/clip` command appears automatically — Hermes reads
`~/.hermes/skills/clip/SKILL.md` and substitutes `${HERMES_SKILL_DIR}`
with the install path.

## Why this is separate from Claude Code / Codex

Both Claude Code (`/plugin marketplace add`) and Codex
(`codex marketplace add`) have native plugin install flows that clone
the repo into their own plugin caches and surface `/clip:clip` (Claude)
or the equivalent slash command. The first `/clip` invocation
auto-runs `bootstrap-venv.sh` if the `.venv` is missing.

Hermes does not have a marketplace yet, but its self-contained
workspace at `~/.hermes/` is the natural install location — no global
config dirs, no `~/.local/share/`, no symlinks. Everything Clippos
writes (config, model cache, rendered MP4s) defaults to paths that
respect the user's chosen output directory; the install itself lives
fully under `~/.hermes/skills/clip/`.

## First /clip run

```text
/clip /absolute/path/to/video.mp4
```

The first invocation will:

1. Auto-download Whisper large-v3 (~3 GB), SpeechBrain ECAPA (~80 MB),
   RetinaFace (~119 MB), RAFT (smaller). Cached after first run.
2. Run the full pipeline: transcribe → diarize → vision → mine → brief
   handoff → score handoff → review → render.
3. Emit rendered MP4s under `~/Documents/Clippos/jobs/<job_id>/renders/`
   (override with `/clip config --output-dir ~/path/to/wherever`).

Expect ~5 min mining + ~1-2 min render per ratio on a 10-minute source
on Apple Silicon M2 Pro. See [Hardware requirements](README.md#hardware-requirements)
for scaling.

## Updating

```bash
cd ~/.hermes/skills/clip
git pull
.venv/bin/pip install -e '.[engine]'   # picks up any pin changes
```

If `bootstrap-venv.sh` ever needs to re-run from scratch:

```bash
rm -rf ~/.hermes/skills/clip/.venv
bash ~/.hermes/skills/clip/scripts/bootstrap-venv.sh
```

## Uninstall

```bash
rm -rf ~/.hermes/skills/clip
rm -f ~/.config/clippos/.env             # if not also using Claude/Codex
rm -rf ~/.cache/clippos                  # model weights (optional, ~3.5 GB)
rm -rf ~/Documents/Clippos               # rendered MP4s (optional, your data)
```

## Troubleshooting

**Python 3.12 not found.** `bootstrap-venv.sh` exits early with brew /
apt / pacman hints. Set `CLIPPOS_BOOTSTRAP_PYTHON=/path/to/python3.12`
to override the discovery if you have a non-PATH install.

**Engine extras failed to install.** Most often a pip cache + uv-strict
metadata edge case. From `~/.hermes/skills/clip/`:
```bash
rm -rf .venv
bash scripts/bootstrap-venv.sh
```

**Check what's wired up:**
```bash
~/.hermes/skills/clip/.venv/bin/python \
  ~/.hermes/skills/clip/scripts/hermes_clip.py preflight
```
Returns JSON with `ready: true/false`, missing engine modules, the
active Python interpreter, and the resolved ffmpeg path.

## Support

- Repo: https://github.com/dylan-buck/Clippos
- Issues: https://github.com/dylan-buck/Clippos/issues
- Hermes: https://github.com/mercurial-tf/hermes
