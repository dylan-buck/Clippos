# Pre-ship fixes

Blockers that must land before the `curl | bash` install path is exposed
publicly. Surfaced during the 2026-04-25 dogfood, where running `/clippos`
on a real YouTube video required ~90 minutes of manual dep-conflict
debugging in a parallel Claude Code session before mining could even
start. None of these are bugs in the engine code itself — they are all
install-path / dep-resolution gaps.

If a user without Anthropic-grade tooling hits any of these, the
"zero-config" promise in the README is broken. Do not ship until the
verification gate at the bottom passes cold on a fresh venv.

---

## Critical (block install on common machines)

### F1. Python version check in the bootstrap script — RESOLVED

**Status.** install.sh was deleted in favor of native plugin
marketplaces per harness (see [README install matrix](../README.md#install)).
The Python version check logic moved to `scripts/bootstrap-venv.sh`,
which every install path invokes (Claude Code lazy on first /clippos,
Codex lazy on first /clippos, Hermes explicitly post-clone). Verified at
`scripts/bootstrap-venv.sh:18-54` — probes each candidate Python with
a `(3,12) <= sys.version_info < (3,13)` check and exits with brew /
apt / pacman hints when nothing qualifies.

**Symptom (historical).** On macOS with the Homebrew default (Python
3.14) and no python3.12 installed, the previous install path proceeded
against an unsupported interpreter, then failed deep inside the pip
resolver on TensorFlow wheels (TF caps at 3.12).

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

**Fix.** Replace the engine block with the resolver-verified pin set
(see `pyproject.toml` for the canonical version — this is a snapshot,
not the source of truth):

```toml
engine = [
    # Pinned to a coexisting set, dogfood-verified 2026-04-25, then
    # bumped to a uv-resolvable set on the same day (see commit 2ac9338).
    # Original 2.3.1/3.3.6 pins worked under warmed pip but failed
    # `uv sync` because whisperx 3.3.6's metadata declares torch>=2.5.1.
    # The current set anchors on pyannote.audio 4.x's torch>=2.8 floor;
    # uv lock --check passes and the lockfile shrank by ~1.4k lines as
    # a result. Loosen these only after re-running BOTH tracks of the
    # verification gate below.
    "torch==2.8.0",
    "torchaudio==2.8.0",
    "torchvision==0.23.0",
    "whisperx==3.8.5",          # earlier 3.3.x had undeclared matplotlib + bad torch metadata
    "transformers>=4.48,<5",    # 5.x dropped compat whisperx still depends on
    "pyannote.audio>=4.0,<5",   # the torch>=2.8 anchor for the rest of the stack
    "speechbrain>=1.0,<2",      # 0.5 vs 1.x EncoderClassifier moved (handled in code)
    "silero-vad>=5",
    "scikit-learn>=1.3",
    "static-ffmpeg>=3.0",
    "opencv-python>=4.9",
    "scenedetect[opencv]>=0.6.4",
    "retina-face>=0.0.17",
    "tf-keras>=2.16",
    "matplotlib>=3.7",          # whisperx imports it without declaring
    "numpy>=2.1",
]
```

Add a unit test that reads `pyproject.toml` and asserts these specific
pins exist, so a future "let me unpin this for flexibility" diff gets
caught at CI rather than on a user's first install.

**Why two separate verification tracks (pip + uv)?** The 2.3.1/3.3.6
pin set in this commit's first iteration shipped a real bug: it
worked in a warmed pip environment but failed fresh `uv sync` because
whisperx 3.3.6's metadata declared `torch>=2.5.1` even though it
imported on 2.3.1. Pip ignored the metadata contradiction; uv didn't.
The verification gate below requires both resolvers to pass cold so a
recurrence of that class of bug shows up at CI rather than at the
user's first install.

---

### F4. `speechbrain` import path moved between 0.5 and 1.x

**Symptom.** `from speechbrain.pretrained import EncoderClassifier`
raises on speechbrain 1.x.

**Status.** Already fixed in `src/clippos/adapters/speechbrain_diarize.py`
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

- `scripts/clippos_skill.py` — new `probe_engine_imports()` plus a
  top-level `ready` field on `config-check`. The `engine_imports`
  block lists every required module's import status with the active
  interpreter path, so a "ready" report now actually means "runnable
  end-to-end". Closes the gap that let the dogfood proceed past
  preflight on a venv missing whisperx.
- `scripts/hermes_clippos.py` — `preflight` propagates engine misses
  into the top-level `missing` list as `engine:<module>` entries with
  an instruction message that names the active interpreter.
- `commands/clippos.md`, `commands/clippos-config.md`, `commands/clippos-package.md`,
  `SKILL.md` — replaced the brittle
  `${CLIPPOS_ROOT:-${HERMES_SKILL_DIR:-${CLAUDE_PLUGIN_ROOT:-$PWD}}}`
  prologue with a robust resolution chain that tries env vars →
  install.sh symlinks → install dir → persisted `~/.config/clippos/.env`
  → `$PWD`. Each candidate is validated by checking for
  `scripts/hermes_clippos.py`. Resolves the Claude Code
  `${CLAUDE_PLUGIN_ROOT}` expansion bug (Anthropic issue #9354) in
  the wild.
- `scripts/clippos_skill.py config-write --root <path>` — new flag that
  persists `CLIPPOS_ROOT` to the config file with validation.
  `scripts/bootstrap-venv.sh` calls it post-install as its last step.
- 370 tests passing, ruff clean.

---

## Verification gate

Do not declare the install path shipped until **both** Track A and
Track B complete cold, without manual intervention, on a Mac with no
prior clippos state. Pip and uv have different resolvers; an
engine-extra set that resolves under one can fail under the other,
and the original 2.3.1 torch / 3.3.6 whisperx pin set was a real
example — it loaded fine in a warmed pip env but was unsolvable under
fresh `uv sync` because whisperx's metadata declared
`torch>=2.5.1` even though it imported on 2.3.1. Pip ignored the
contradiction; uv didn't. Verifying both keeps that class of bug from
recurring.

### Track A — pip via `bootstrap-venv.sh` (the install path each harness invokes)

```bash
# 1. Wipe any prior install + venv (Hermes, Claude, Codex caches; legacy
#    install dir from when there was a top-level install.sh).
rm -rf ~/.local/share/clippos /tmp/clippos-test-venv \
       ~/.hermes/skills/clippos \
       ~/.claude/plugins/cache/Clippos \
       ~/.codex/plugins/cache/Clippos \
       ~/.config/clippos

# 2. Simulate the Hermes install path (the most explicit; Claude / Codex
#    marketplace adds clone into versioned cache dirs we don't control).
git clone https://github.com/dylan-buck/Clippos ~/.hermes/skills/clippos
bash ~/.hermes/skills/clippos/scripts/bootstrap-venv.sh

# 3. Smoke-test engine imports without touching code
~/.hermes/skills/clippos/.venv/bin/python -c "
import whisperx, speechbrain, silero_vad, cv2, retinaface, \
       torch, torchvision, scenedetect, matplotlib
print('engine ok:', torch.__version__, whisperx.__version__)
"

# 4. Run preflight via the published skill path
~/.hermes/skills/clippos/.venv/bin/python \
  ~/.hermes/skills/clippos/scripts/hermes_clippos.py preflight
# Expect ready=true, no engine:* entries in `missing`.

# 5. Real /clippos end-to-end on a short test video (5–10 min)
~/.hermes/skills/clippos/.venv/bin/python \
  ~/.hermes/skills/clippos/scripts/hermes_clippos.py advance \
  --source /absolute/path/to/short-test.mp4
# Expect: workspace ready, mine completes, scoring handoff emitted.
```

### Track B — `uv sync` (catches resolver-strict pin contradictions)

`uv sync` enforces declared metadata constraints that pip will silently
ignore. Run this from a fresh checkout in a clean tmpdir to confirm
the engine extras resolve cleanly without tapping any pre-warmed
caches:

```bash
# 1. Fresh checkout into a clean tmpdir (no prior .venv to warm the resolver)
rm -rf /tmp/clippos-uv-test
git clone https://github.com/dylan-buck/Clippos /tmp/clippos-uv-test
cd /tmp/clippos-uv-test

# 2. Resolve + sync engine extras with uv
uv sync --extra engine

# 3. Same engine-import smoke test as Track A
.venv/bin/python -c "
import whisperx, speechbrain, silero_vad, cv2, retinaface, \
       torch, torchvision, scenedetect, matplotlib
print('engine ok:', torch.__version__, whisperx.__version__)
"

# 4. Confirm the lockfile is consistent with pyproject.toml — uv lock
#    --check exits non-zero if any pin moved or any transitive resolution
#    drifted since the lockfile was last regenerated.
uv lock --check
```

Both tracks must pass cold. If Track A passes but Track B fails, the
pin set has metadata contradictions even when it works in pip — fix
the pins (see F3) before merging. If Track B passes but Track A fails,
something has diverged in `bootstrap-venv.sh`'s pip path; investigate
before shipping.

If any step in either track fails, the install path is not yet ready
for public release. Add the failure mode to this doc and fix it before
retrying.

---

## Cosmetic noise (low priority but worth fixing before public ship)

### N1. Duplicate-class warnings from multiple bundled libav — RESOLVED

**Status.** Shipped in commit `0c66755` via Option 2: the
`hermes_clippos.py:_run_cli_stage` live-stderr passthrough now filters
the multi-line `objc[*]: Class … is implemented in both …` block,
preserving real diagnostic content. Test:
`tests/scripts/test_hermes_clippos.py::test_is_objc_dylib_warning_matches_full_warning_block`.

Option 3 (standardize on a single ffmpeg) is still the right
long-term architectural answer but requires rebuilding wheels —
deferred.

---

## Followups (lower priority, log here so they don't get lost)

- **whisperx vs transformers 5.x.** Pin `transformers<5` is a stop-gap;
  whisperx maintenance may catch up. Re-evaluate when whisperx 4.x
  ships or when 3.x cuts a release that drops the upper bound.
- **Linux / Windows install paths.** All bugs above are observed on
  macOS arm64. The same pins should work on Linux x86_64 (TF wheels
  available, torch 2.8 wheels available), but neither has been
  dogfood-verified. Add both to the verification gate when a Linux
  dev box is available.
- **CUDA torch on Linux.** The current `torch==2.8.0` pulls the CPU
  wheel by default on Linux. Users with NVIDIA GPUs will want a
  CUDA-suffixed wheel (e.g. `torch==2.8.0+cu124`). Document the
  override pattern (`pip install torch==2.8.0+cu124 --index-url
  https://download.pytorch.org/whl/cu124`) rather than trying to
  detect it in `bootstrap-venv.sh`.
