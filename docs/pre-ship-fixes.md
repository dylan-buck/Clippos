# Pre-ship fixes

Blockers that must land before the `curl | bash` install path is exposed
publicly. Surfaced during the 2026-04-25 dogfood, where running `/clip`
on a real YouTube video required ~90 minutes of manual dep-conflict
debugging in a parallel Claude Code session before mining could even
start. None of these are bugs in the engine code itself — they are all
install-path / dep-resolution gaps.

If a user without Anthropic-grade tooling hits any of these, the
"zero-config" promise in the README is broken. Do not ship until the
verification gate at the bottom passes cold on a fresh venv.

---

## Critical (block install on common machines)

### F1. `install.sh find_python` accepts wrong Python version

**Symptom.** On macOS with the current Homebrew default (Python 3.14)
and no python3.12 installed, `install.sh` proceeds against an
unsupported interpreter, then fails deep inside the pip resolver on
TensorFlow wheels (TF caps at 3.12).

**Root cause.** `find_python` only checks `command -v` exists, not
version. The error string says "Python 3.12+ is required" but the
script doesn't enforce it.

**Fix.** Replace `find_python` so it probes each candidate with a real
version check:

```bash
find_python() {
  if [ -n "${CLIPPER_BOOTSTRAP_PYTHON:-}" ]; then
    if "$CLIPPER_BOOTSTRAP_PYTHON" -c 'import sys; sys.exit(0 if (3,12) <= sys.version_info < (3,13) else 1)' >/dev/null 2>&1; then
      printf '%s\n' "$CLIPPER_BOOTSTRAP_PYTHON"; return
    fi
    printf 'CLIPPER_BOOTSTRAP_PYTHON=%s does not satisfy 3.12.x\n' \
      "$CLIPPER_BOOTSTRAP_PYTHON" >&2
    exit 1
  fi
  for candidate in python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && \
       "$candidate" -c 'import sys; sys.exit(0 if (3,12) <= sys.version_info < (3,13) else 1)' >/dev/null 2>&1; then
      printf '%s\n' "$candidate"; return
    fi
  done
  printf 'Need Python 3.12 (TensorFlow wheels cap at 3.12, our pyproject pins <3.13).\n' >&2
  printf 'On macOS: brew install python@3.12\n' >&2
  printf 'On Debian/Ubuntu: sudo apt install python3.12 python3.12-venv\n' >&2
  exit 1
}
```

Update the calling site to drop the redundant `command -v` check (the
new `find_python` already exits non-zero with a clear message when
nothing qualifies).

---

### F2. `pyproject.toml` Python pin too loose

**Symptom.** `requires-python = ">=3.12"` plus the TF dependency cap
(`<=3.12`) means only Python 3.12 can ever resolve. Users on 3.13/3.14
get past `pip install` validation, then fail on opaque TF wheel errors.

**Fix.**

```toml
[project]
requires-python = ">=3.12,<3.13"
```

This makes `pip install` reject incompatible interpreters before
attempting to resolve TF — the failure becomes local and obvious.

---

### F3. Engine extras need exact pins, not floors

**Symptom.** Cascade of import-time crashes after a clean `pip install
-e '.[engine]'`:

- `pip` resolves `torch==2.11`, `torchaudio==2.11`.
- `pyannote.audio==3.3.2` imports `torchaudio.AudioMetaData` (removed
  in torchaudio 2.4) → `ImportError`.
- `whisperx` imports `pyannote` unconditionally, so even the
  speechbrain-default diarizer path is blocked.

**Root cause.** Engine extras are floor-pinned (`torch>=2.2`,
`pyannote.audio>=3.1`). Pip picks newest-resolving versions, which is
how you end up with torch 2.11 and pyannote 3.3 trying to coexist.

**Fix.** Replace the engine block with the dogfood-verified pin set:

```toml
engine = [
    # Pinned to a coexisting set, dogfood-verified 2026-04-25.
    # Loosen these only after re-running the smoke test below.
    "torch==2.3.1",
    "torchaudio==2.3.1",
    "torchvision==0.18.1",
    "whisperx==3.3.6",          # 3.4.x adds an undeclared matplotlib dep
    "transformers>=4.40,<5",    # 5.x removed compat whisperx 3.x relies on
    "pyannote.audio>=3.3,<4",   # 4.x needs torch>=2.8 — incompatible cascade
    "speechbrain>=1.0,<2",      # 0.5 vs 1.x EncoderClassifier moved (handled in code)
    "silero-vad>=5",
    "scikit-learn>=1.3",
    "static-ffmpeg>=3.0",
    "opencv-python>=4.9",
    "scenedetect[opencv]>=0.6.4",
    "retina-face>=0.0.17",
    "tf-keras>=2.16",
    "matplotlib>=3.7",          # whisperx imports it without declaring
    "numpy>=1.26",
]
```

Add a unit test that reads `pyproject.toml` and asserts these specific
pins exist, so a future "let me unpin this for flexibility" diff gets
caught at CI rather than on a user's first install.

---

### F4. `speechbrain` import path moved between 0.5 and 1.x

**Symptom.** `from speechbrain.pretrained import EncoderClassifier`
raises on speechbrain 1.x.

**Status.** Already fixed in `src/clipper/adapters/speechbrain_diarize.py`
during the 2026-04-25 dogfood — added a try/except that prefers the
1.x path (`speechbrain.inference.speaker`) and falls back to the 0.5
path (`speechbrain.pretrained`).

**Followup.** Add a test that asserts the dual-path fallback works
when only the legacy module is importable (monkeypatch the new path to
raise `ImportError`). Otherwise a future refactor that drops the
fallback would silently break 0.5-only environments without anything
catching it.

---

## Done in parallel during dogfood (do not duplicate)

The following landed in this session before the dep conflicts were
discovered. Pull latest before editing the files above.

- `scripts/clip_skill.py` — new `probe_engine_imports()` plus a
  top-level `ready` field on `config-check`. The `engine_imports`
  block lists every required module's import status with the active
  interpreter path, so a "ready" report now actually means "runnable
  end-to-end". Closes the gap that let the dogfood proceed past
  preflight on a venv missing whisperx.
- `scripts/hermes_clip.py` — `preflight` propagates engine misses
  into the top-level `missing` list as `engine:<module>` entries with
  an instruction message that names the active interpreter.
- `commands/clip.md`, `commands/clip-config.md`, `commands/clip-package.md`,
  `SKILL.md` — replaced the brittle
  `${CLIPPER_ROOT:-${HERMES_SKILL_DIR:-${CLAUDE_PLUGIN_ROOT:-$PWD}}}`
  prologue with a robust resolution chain that tries env vars →
  install.sh symlinks → install dir → persisted `~/.config/clipper-tool/.env`
  → `$PWD`. Each candidate is validated by checking for
  `scripts/hermes_clip.py`. Resolves the Claude Code
  `${CLAUDE_PLUGIN_ROOT}` expansion bug (Anthropic issue #9354) in
  the wild.
- `scripts/clip_skill.py config-write --root <path>` — new flag that
  persists `CLIPPER_ROOT` to the config file with validation. `install.sh`
  now calls it post-install via `persist_clipper_root`.
- 370 tests passing, ruff clean.

---

## Verification gate

Do not declare the install path shipped until this completes cold,
without manual intervention, on a Mac with no prior clipper-tool state:

```bash
# 1. Wipe any prior install + venv
rm -rf ~/.local/share/clipping-tool /tmp/clipper-test-venv \
       ~/.hermes/skills/clip ~/.claude/skills/clip ~/.codex/skills/clip

# 2. Run the one-liner exactly as a user would
curl -fsSL https://raw.githubusercontent.com/dylan-buck/clipping-tool/main/install.sh | bash

# 3. Smoke-test engine imports without touching code
~/.local/share/clipping-tool/.venv/bin/python -c "
import whisperx, speechbrain, silero_vad, cv2, retinaface, \
       torch, torchvision, scenedetect, matplotlib
print('engine ok:', torch.__version__, whisperx.__version__)
"

# 4. Run preflight via the published skill path
~/.local/share/clipping-tool/.venv/bin/python \
  ~/.local/share/clipping-tool/scripts/hermes_clip.py preflight
# Expect ready=true, no engine:* entries in `missing`.

# 5. Real /clip end-to-end on a short test video (5–10 min)
~/.local/share/clipping-tool/.venv/bin/python \
  ~/.local/share/clipping-tool/scripts/hermes_clip.py advance \
  --source /absolute/path/to/short-test.mp4
# Expect: workspace ready, mine completes, scoring handoff emitted.
```

If any step fails, the install path is not yet ready for public
release. Add the failure mode to this doc and fix it before retrying.

---

## Cosmetic noise (low priority but worth fixing before public ship)

### N1. Duplicate-class warnings from multiple bundled libav

**Symptom.** Every clipper run on macOS prints a wall of:

```
objc[xxxxx]: Class AVFFrameReceiver is implemented in both
  /opt/homebrew/Cellar/ffmpeg@5/.../libavdevice.59.7.100.dylib (0x...)
  and /Users/.../site-packages/cv2/.dylibs/libavdevice.61.3.100.dylib (0x...)
  One of the two will be used. Which one is undefined.
```

**Root cause.** `cv2`, `av` (pyannote dep), `static-ffmpeg`, AND the
Homebrew system `ffmpeg` each ship their own copy of libav*. macOS
loads all of them, sees the duplicate Objective-C classes, and warns.

**Impact.** Cosmetic. No functional bug, but it pollutes user-visible
stderr and makes real errors harder to spot.

**Possible fixes (pick one):**

1. Set `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` in
   `clipper/__init__.py` alongside the existing `TF_USE_LEGACY_KERAS`
   set — silences the warnings at the cost of disabling a safety
   check we don't actually use.
2. Filter the `objc[*]` lines out of the live stderr passthrough in
   `hermes_clip.py:_run_cli_stage` so users never see them.
3. Standardize on a single ffmpeg (the vendored static-ffmpeg from
   `4c0e090`) and make `cv2` / `av` use the same shared libs. This is
   the right fix architecturally but requires rebuilding wheels.

Option 2 is the lowest-risk and most user-visible win. Option 3 is
the right long-term answer but costs real engineering time.

---

## Followups (lower priority, log here so they don't get lost)

- **whisperx vs transformers 5.x.** Pin `transformers<5` is a stop-gap;
  whisperx maintenance may catch up. Re-evaluate when whisperx 3.5+
  ships.
- **pyannote.audio 4.x track.** Requires `torch>=2.8`, which would
  cascade through the entire pin set. Defer until torch 2.8+ is the
  natural floor for everything else.
- **Linux / Windows install paths.** All bugs above are observed on
  macOS arm64. The same pins should work on Linux x86_64 (TF wheels
  available, torch wheels available), but neither has been
  dogfood-verified. Add to the verification gate when a Linux dev box
  is available.
- **CUDA torch on Linux.** The current `torch==2.3.1` pulls the CPU
  wheel by default on Linux. Users with NVIDIA GPUs will want
  `torch==2.3.1+cu118` or similar. Document the override pattern
  (`pip install torch==2.3.1+cu118 --index-url https://download.pytorch.org/whl/cu118`)
  rather than trying to detect it in install.sh.
