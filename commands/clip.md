---
description: Analyze a video link, local path, or attached video file and render high-potential social clips with captions and crops.
argument-hint: '<video link|path|attached file> [--ratios 9:16,1:1,16:9] [--clips 5] [--min-score 0.70]'
allowed-tools: [Bash, Read, Write, AskUserQuestion]
---

Invoke the `clip` skill with the user's arguments: $ARGUMENTS

Resolve `CLIPPOS_ROOT` with the prologue documented in `SKILL.md` (env var
> `HERMES_SKILL_DIR` > `CLAUDE_PLUGIN_ROOT` > `~/.hermes/skills/clip` >
`~/.claude/skills/clip` > `~/.codex/skills/clip` >
`~/.local/share/clippos` > `CLIPPOS_ROOT` line in
`~/.config/clippos/.env` > `$PWD`). Each candidate must contain
`scripts/hermes_clip.py`. Then resolve `CLIPPOS_PYTHON` to
`$CLIPPOS_ROOT/.venv/bin/python` when executable, otherwise `python3`.

Run the full loop: prepare the source, mine candidates, score every candidate
with the harness model, write `scoring-response.json`, review, approve selected
clips, render final MP4s, and return the rendered output paths.
