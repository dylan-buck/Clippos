---
description: Configure the local clip skill environment, including output directory, ratios, clip count, score threshold, and Hugging Face token status.
argument-hint: '[--output-dir ~/Documents/Clippos] [--ratios 9:16,1:1,16:9] [--clips 5] [--hf-token hf_...]'
allowed-tools: [Bash, Read, Write, AskUserQuestion]
---

Invoke the `clip` skill's configuration workflow with the user's arguments:
$ARGUMENTS

Resolve `CLIPPOS_ROOT` with the prologue documented in `SKILL.md` (env
var > `HERMES_SKILL_DIR` > `CLAUDE_PLUGIN_ROOT` > `~/.hermes/skills/clip`
> `~/.claude/skills/clip` > `~/.codex/skills/clip` > newest match in
`~/.claude/plugins/cache/<marketplace>/clip/<sha>` /
`~/.codex/plugins/cache/<marketplace>/clip/<sha>` > `CLIPPOS_ROOT`
line in `~/.config/clippos/.env` > `$PWD`). Each candidate must contain
`scripts/hermes_clip.py`. If `$CLIPPOS_ROOT/.venv` does not exist, run
`bash $CLIPPOS_ROOT/scripts/bootstrap-venv.sh` once to create it. Then
resolve `CLIPPOS_PYTHON` to `$CLIPPOS_ROOT/.venv/bin/python` when
executable, otherwise `python3`.

Run `"$CLIPPOS_PYTHON" "$CLIPPOS_ROOT/scripts/clip_skill.py" config-check`
first. The payload's top-level `ready` field reflects bins, render path, and
engine extras together — anything other than `true` means the run will fail
somewhere. If `engine_imports.missing_required` is non-empty, the active
interpreter lacks engine extras; tell the user to either install them in
that interpreter or point `CLIPPOS_PYTHON` at one that has them.

If the user provided defaults, write them with
`"$CLIPPOS_PYTHON" "$CLIPPOS_ROOT/scripts/clip_skill.py" config-write ...`.
For dev checkouts that lack a persisted root, also pass `--root
"$CLIPPOS_ROOT"` once so the prologue resolves cleanly next time. If
required values are missing, ask only for the missing value needed next.
