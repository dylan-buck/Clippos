#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${CLIPPER_REPO_URL:-https://github.com/dylan-buck/clipping-tool.git}"
REF="${CLIPPER_REF:-main}"
INSTALL_DIR="${CLIPPER_INSTALL_DIR:-$HOME/.local/share/clipping-tool}"
HARNESS="${CLIPPER_HARNESS:-all}"
EXTRAS="${CLIPPER_EXTRAS:-engine}"

log() {
  printf '[clip install] %s\n' "$*"
}

warn() {
  printf '[clip install] warning: %s\n' "$*" >&2
}

find_python() {
  if [ -n "${CLIPPER_BOOTSTRAP_PYTHON:-}" ]; then
    printf '%s\n' "$CLIPPER_BOOTSTRAP_PYTHON"
    return
  fi
  for candidate in python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  printf 'python3\n'
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

persist_clipper_root() {
  # Pin CLIPPER_ROOT in the config file so the skill prologue can resolve
  # the install dir even when the harness does not propagate
  # CLAUDE_PLUGIN_ROOT / HERMES_SKILL_DIR (e.g. Claude Code issue #9354).
  log "Persisting CLIPPER_ROOT to ~/.config/clipper-tool/.env"
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
      printf 'Unsupported CLIPPER_HARNESS=%s (use all, hermes, claude, codex, or none)\n' "$HARNESS" >&2
      exit 2
      ;;
  esac
}

main() {
  if ! command -v git >/dev/null 2>&1; then
    printf 'git is required to install clipping-tool\n' >&2
    exit 1
  fi

  local python_bin
  python_bin="$(find_python)"
  if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'Python 3.12+ is required to install clipping-tool\n' >&2
    exit 1
  fi

  checkout_repo
  install_python_env "$python_bin"
  persist_clipper_root
  install_harness_links

  log "Installed clip skill at $INSTALL_DIR"
  log "First /clip run downloads ~3.5 GB of model weights (Whisper large-v3,"
  log "SpeechBrain ECAPA, RetinaFace, RAFT) and caches them locally."
  log "16 GB RAM is the practical floor; expect 3-5 min mining + 1-2 min"
  log "render per ratio on a 10-min video on Apple Silicon. The fan will"
  log "spin up during vision — that's normal."
  log "Try: /clip status"
}

main "$@"
