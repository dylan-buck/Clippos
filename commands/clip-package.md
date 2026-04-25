---
description: Generate a publish pack (titles, thumbnails, caption, hashtags, hooks) for every approved, rendered clip in a workspace.
argument-hint: '<workspace path> (defaults to the most recent skill-jobs workspace)'
allowed-tools: [Bash, Read, Write]
---

Invoke the `clip` skill's packaging workflow for: $ARGUMENTS

Resolve `CLIPPOS_ROOT` with the prologue documented in `SKILL.md` (env var
> `HERMES_SKILL_DIR` > `CLAUDE_PLUGIN_ROOT` > `~/.hermes/skills/clip` >
`~/.claude/skills/clip` > `~/.codex/skills/clip` >
`~/.local/share/clippos` > `CLIPPOS_ROOT` line in
`~/.config/clippos/.env` > `$PWD`). Each candidate must contain
`scripts/hermes_clip.py`. Then resolve `CLIPPOS_PYTHON` to
`$CLIPPOS_ROOT/.venv/bin/python` when executable, otherwise `python3`.

Resolve the target workspace (either the argument, or the most recent
`skill-jobs/*/jobs/<job_id>/` directory under the configured output dir). Then:

1. Run `package-prompt` against the workspace to emit `package-request.json`.
2. Read `package-request.json`, follow its `package_prompt` and
   `response_schema`, and write `package-response.json` beside the request.
3. Run `package-save` to validate the response, fan per-clip `package.json`
   files into `renders/<clip_id>/`, and write the summary `package-report.json`.
4. Return the per-clip pack paths plus the rendered MP4 paths they sit next to.
