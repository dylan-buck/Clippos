#!/usr/bin/env bash
# scripts/bootstrap-venv.sh
#
# Idempotent first-run setup: create a .venv at the Clippos install root
# and pip-install the engine extras. Called automatically by the SKILL
# prologue on the first /clippos invocation when no .venv exists.
#
# Why this is needed: native plugin managers (Claude Code's
# `/plugin marketplace add`, Codex's equivalent) clone the repo but do
# not run pip — there's no install hook to declare the heavy ML deps.
# Without this, the first /clippos call would fail with engine_imports
# missing (whisperx, torch, retina-face, etc.). This script closes that
# gap with one self-contained step.
#
# Hermes's HERMES_SETUP.md calls this script directly post-clone, so
# Hermes users don't pay the cost on first /clippos invocation.
#
# Cost: ~5 min for pip downloads (~700 MB of wheels). Models themselves
# (~3.5 GB Whisper + RetinaFace + RAFT) download lazily on first /clippos.

set -euo pipefail

CLIPPOS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Idempotent: skip if already bootstrapped. Cheap to call from the
# SKILL prologue on every invocation.
if [ -d "$CLIPPOS_ROOT/.venv" ]; then
  exit 0
fi

python_in_supported_range() {
  # The pyproject pin is >=3.12,<3.13 because TensorFlow wheels (pulled
  # transitively via retina-face) cap at 3.12 — anything newer fails
  # mid-resolve with a confusing TF error far from the real cause.
  "$1" -c 'import sys; sys.exit(0 if (3,12) <= sys.version_info < (3,13) else 1)' \
    >/dev/null 2>&1
}

find_python() {
  if [ -n "${CLIPPOS_BOOTSTRAP_PYTHON:-}" ]; then
    if python_in_supported_range "$CLIPPOS_BOOTSTRAP_PYTHON"; then
      printf '%s\n' "$CLIPPOS_BOOTSTRAP_PYTHON"
      return
    fi
    printf '[clippos] CLIPPOS_BOOTSTRAP_PYTHON=%s does not satisfy 3.12.x\n' \
      "$CLIPPOS_BOOTSTRAP_PYTHON" >&2
    exit 1
  fi
  for candidate in python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 \
      && python_in_supported_range "$candidate"; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  printf '[clippos] Need Python 3.12.x on PATH.\n' >&2
  printf '[clippos] TensorFlow wheels (pulled via retina-face) cap at 3.12,\n' >&2
  printf '[clippos] and our pyproject pins requires-python = ">=3.12,<3.13".\n' >&2
  printf '\n' >&2
  printf '[clippos]   macOS:        brew install python@3.12\n' >&2
  printf '[clippos]   Debian/Ubuntu: sudo apt install python3.12 python3.12-venv\n' >&2
  printf '[clippos]   Arch:          sudo pacman -S python (currently 3.12.x)\n' >&2
  printf '\n' >&2
  printf '[clippos] Or set CLIPPOS_BOOTSTRAP_PYTHON=/path/to/python3.12 and re-run.\n' >&2
  exit 1
}

PY="$(find_python)"

printf '[clippos] First-run setup: creating .venv at %s\n' "$CLIPPOS_ROOT/.venv" >&2
printf '[clippos] This downloads ~700 MB of pip wheels and takes 3-7 min.\n' >&2
printf '[clippos] Subsequent /clippos calls reuse the cached .venv.\n' >&2

"$PY" -m venv "$CLIPPOS_ROOT/.venv"
"$CLIPPOS_ROOT/.venv/bin/python" -m pip install --upgrade --quiet pip setuptools wheel
"$CLIPPOS_ROOT/.venv/bin/python" -m pip install -e "$CLIPPOS_ROOT[engine]"

# Persist CLIPPOS_ROOT so the SKILL prologue can resolve it across
# harnesses without relying on env-var substitution that some harnesses
# (Claude Code, Anthropic issue #9354) don't always perform.
"$CLIPPOS_ROOT/.venv/bin/python" "$CLIPPOS_ROOT/scripts/clippos_skill.py" \
  config-write --root "$CLIPPOS_ROOT"

printf '[clippos] Setup complete. Engine extras installed.\n' >&2
printf '[clippos] First /clippos will additionally download ~3.5 GB of model weights\n' >&2
printf '[clippos] (Whisper large-v3, SpeechBrain ECAPA, RetinaFace, RAFT) on first use.\n' >&2
