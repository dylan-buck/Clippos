---
description: Configure the local clip skill environment, including output directory, ratios, clip count, score threshold, and Hugging Face token status.
argument-hint: '[--output-dir ~/Documents/ClipperTool] [--ratios 9:16,1:1,16:9] [--clips 5] [--hf-token hf_...]'
allowed-tools: [Bash, Read, Write, AskUserQuestion]
---

Invoke the `clip` skill's configuration workflow with the user's arguments:
$ARGUMENTS

Resolve `CLIPPER_ROOT` with the prologue documented in `SKILL.md` (env var
> `HERMES_SKILL_DIR` > `CLAUDE_PLUGIN_ROOT` > `~/.hermes/skills/clip` >
`~/.claude/skills/clip` > `~/.codex/skills/clip` >
`~/.local/share/clipping-tool` > `CLIPPER_ROOT` line in
`~/.config/clipper-tool/.env` > `$PWD`). Each candidate must contain
`scripts/hermes_clip.py`. Then resolve `CLIPPER_PYTHON` to
`$CLIPPER_ROOT/.venv/bin/python` when executable, otherwise `python3`.

Run `"$CLIPPER_PYTHON" "$CLIPPER_ROOT/scripts/clip_skill.py" config-check`
first. The payload's top-level `ready` field reflects bins, render path, and
engine extras together — anything other than `true` means the run will fail
somewhere. If `engine_imports.missing_required` is non-empty, the active
interpreter lacks engine extras; tell the user to either install them in
that interpreter or point `CLIPPER_PYTHON` at one that has them.

If the user provided defaults, write them with
`"$CLIPPER_PYTHON" "$CLIPPER_ROOT/scripts/clip_skill.py" config-write ...`.
For dev checkouts that lack a persisted root, also pass `--root
"$CLIPPER_ROOT"` once so the prologue resolves cleanly next time. If
required values are missing, ask only for the missing value needed next.
