---
description: Generate a publish pack (titles, thumbnails, caption, hashtags, hooks) for every approved, rendered clip in a workspace.
argument-hint: '<workspace path> (defaults to the most recent skill-jobs workspace)'
allowed-tools: [Bash, Read, Write]
---

Invoke the `clip` skill's packaging workflow for: $ARGUMENTS

Resolve the target workspace (either the argument, or the most recent
`skill-jobs/*/jobs/<job_id>/` directory under the configured output dir). Then:

1. Run `package-prompt` against the workspace to emit `package-request.json`.
2. Read `package-request.json`, follow its `package_prompt` and
   `response_schema`, and write `package-response.json` beside the request.
3. Run `package-save` to validate the response, fan per-clip `package.json`
   files into `renders/<clip_id>/`, and write the summary `package-report.json`.
4. Return the per-clip pack paths plus the rendered MP4 paths they sit next to.
