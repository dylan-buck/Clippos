---
description: Configure the local clip skill environment, including output directory, ratios, clip count, score threshold, and Hugging Face token status.
argument-hint: '[--output-dir ~/Documents/ClipperTool] [--ratios 9:16,1:1,16:9] [--clips 3] [--hf-token hf_...]'
allowed-tools: [Bash, Read, Write, AskUserQuestion]
---

Invoke the `clip` skill's configuration workflow with the user's arguments:
$ARGUMENTS

Resolve `CLIPPER_ROOT` to
`${CLIPPER_ROOT:-${HERMES_SKILL_DIR:-${CLAUDE_PLUGIN_ROOT:-$PWD}}}` and
`CLIPPER_PYTHON` to `$CLIPPER_ROOT/.venv/bin/python` when executable,
otherwise `python3`. Run
`"$CLIPPER_PYTHON" "$CLIPPER_ROOT/scripts/clip_skill.py" config-check` first.
If the user provided defaults, write them with
`"$CLIPPER_PYTHON" "$CLIPPER_ROOT/scripts/clip_skill.py" config-write ...`.
If required values are missing, ask only for the missing value needed next.
