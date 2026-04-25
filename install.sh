#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${CLIPPOS_REPO_URL:-https://github.com/dylan-buck/Clippos.git}"
REF="${CLIPPOS_REF:-main}"
INSTALL_DIR="${CLIPPOS_INSTALL_DIR:-$HOME/.local/share/clippos}"
HARNESS="${CLIPPOS_HARNESS:-all}"
EXTRAS="${CLIPPOS_EXTRAS:-engine}"

log() {
  printf '[clip install] %s\n' "$*"
}

warn() {
  printf '[clip install] warning: %s\n' "$*" >&2
}

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
    printf 'CLIPPOS_BOOTSTRAP_PYTHON=%s does not satisfy 3.12.x\n' \
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
  printf 'No Python 3.12.x found on PATH.\n' >&2
  printf 'TensorFlow wheels (pulled via retina-face) cap at 3.12, and our\n' >&2
  printf 'pyproject pins requires-python = ">=3.12,<3.13", so 3.13/3.14 also\n' >&2
  printf 'fail to install.\n' >&2
  printf '\n' >&2
  printf '  macOS:        brew install python@3.12\n' >&2
  printf '  Debian/Ubuntu: sudo apt install python3.12 python3.12-venv\n' >&2
  printf '  Arch:          sudo pacman -S python (currently 3.12.x)\n' >&2
  printf '\n' >&2
  printf 'Or set CLIPPOS_BOOTSTRAP_PYTHON=/path/to/python3.12 and re-run.\n' >&2
  exit 1
}

checkout_repo() {
  if [ -d "$INSTALL_DIR/.git" ]; then
    if git -C "$INSTALL_DIR" diff --quiet && git -C "$INSTALL_DIR" diff --cached --quiet; then
      log "Updating $INSTALL_DIR to $REF"
      git -C "$INSTALL_DIR" fetch --depth 1 origin "$REF"
      git -C "$INSTALL_DIR" checkout -q FETCH_HEAD
    else
      warn "$INSTALL_DIR has local changes; leaving checkout untouched"
    fi
    return
  fi

  if [ -e "$INSTALL_DIR" ]; then
    warn "$INSTALL_DIR exists and is not a git checkout; using it as-is"
    return
  fi

  log "Cloning $REPO_URL into $INSTALL_DIR"
  mkdir -p "$(dirname "$INSTALL_DIR")"
  git clone --depth 1 --branch "$REF" "$REPO_URL" "$INSTALL_DIR"
}

install_python_env() {
  local python_bin="$1"
  log "Creating virtualenv"
  "$python_bin" -m venv "$INSTALL_DIR/.venv"
  "$INSTALL_DIR/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
  if [ "$EXTRAS" = "none" ]; then
    "$INSTALL_DIR/.venv/bin/python" -m pip install -e "$INSTALL_DIR"
  else
    "$INSTALL_DIR/.venv/bin/python" -m pip install -e "$INSTALL_DIR[$EXTRAS]"
  fi
}

persist_clippos_root() {
  # Pin CLIPPOS_ROOT in the config file so the skill prologue can resolve
  # the install dir even when the harness does not propagate
  # CLAUDE_PLUGIN_ROOT / HERMES_SKILL_DIR (e.g. Claude Code issue #9354).
  log "Persisting CLIPPOS_ROOT to ~/.config/clippos/.env"
  "$INSTALL_DIR/.venv/bin/python" "$INSTALL_DIR/scripts/clip_skill.py" \
    config-write --root "$INSTALL_DIR"
}

link_skill() {
  local target="$1"
  mkdir -p "$(dirname "$target")"
  if [ -L "$target" ] || [ ! -e "$target" ]; then
    ln -sfn "$INSTALL_DIR" "$target"
    log "Linked $target"
  else
    warn "$target already exists and is not a symlink; leaving it untouched"
  fi
}

install_harness_links() {
  case "$HARNESS" in
    all)
      link_skill "$HOME/.hermes/skills/clip"
      link_skill "$HOME/.claude/skills/clip"
      link_skill "$HOME/.codex/skills/clip"
      ;;
    hermes)
      link_skill "$HOME/.hermes/skills/clip"
      ;;
    claude)
      link_skill "$HOME/.claude/skills/clip"
      ;;
    codex)
      link_skill "$HOME/.codex/skills/clip"
      ;;
    none)
      log "Skipping harness links"
      ;;
    *)
      printf 'Unsupported CLIPPOS_HARNESS=%s (use all, hermes, claude, codex, or none)\n' "$HARNESS" >&2
      exit 2
      ;;
  esac
}

main() {
  if ! command -v git >/dev/null 2>&1; then
    printf 'git is required to install Clippos\n' >&2
    exit 1
  fi

  local python_bin
  python_bin="$(find_python)"
  # find_python exits non-zero with a clear message if nothing qualifies;
  # the redundant command -v check the original script had is gone.

  checkout_repo
  install_python_env "$python_bin"
  persist_clippos_root
  install_harness_links

  log "Installed Clippos at $INSTALL_DIR"
  log "First /clip run downloads ~3.5 GB of model weights (Whisper large-v3,"
  log "SpeechBrain ECAPA, RetinaFace, RAFT) and caches them locally."
  log "16 GB RAM is the practical floor; expect 3-5 min mining + 1-2 min"
  log "render per ratio on a 10-min video on Apple Silicon. The fan will"
  log "spin up during vision — that's normal."
  log "Try: /clip status"
}

main "$@"
