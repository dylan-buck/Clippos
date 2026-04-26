# Package Handoff Contract

The packaging stage turns every approved, rendered clip into a ready-to-paste
publish pack (titles, thumbnail overlay lines, social caption, hashtags,
opening-line hooks). It mirrors the scoring handoff so the clipper itself
never calls an LLM — the surrounding agent harness is the author, the clipper
is the enforcer and persister.

## Artifacts

All artifacts live beside the existing stage outputs under
`<output_dir>/jobs/<job_id>/`:

```
jobs/<job_id>/
  package-request.json          # written by: clippos_skill.py package-prompt
  package-response.json         # written by: the harness model
  package-report.json           # written by: clippos_skill.py package-save
  renders/<clip_id>/
    package.json                # per-clip publish pack (next to the MP4s)
```

`package-request.json` carries the canonical `package_prompt`, a strict JSON
`response_schema`, and one `PackageBrief` per approved candidate. Briefs reuse
the scoring-request `clip_hash` so the response is matched by hash (not by
list index) — misordered responses fail validation rather than silently
corrupting output.

`package-response.json` is produced by the harness; it must satisfy
`PackageResponse` (`prompt_version`, `job_id`, list of `PublishPack`). The
`prompt_version` and `job_id` are checked against the request before any pack
is written.

`package-report.json` is the workspace-level summary:

```json
{
  "job_id": "<job>",
  "video_path": "/abs/path/input.mp4",
  "packs": [
    {"clip_id": "clip-a", "pack_path": "renders/clip-a/package.json"}
  ]
}
```

## `PublishPack`

One `PublishPack` per approved clip:

- `clip_id`, `clip_hash` — echoed from the request; mismatches are rejected.
- `titles` — at least 5 entries, each ≤ 80 characters. Distinct angles /
  framings; the harness is explicitly told not to reuse `title_hint` verbatim.
- `thumbnail_texts` — at least 3 entries, each ≤ 28 characters. Intended for
  high-contrast overlay use; bias toward all-caps short phrases.
- `social_caption` — single string ≤ 500 characters; 2–4 sentences. No
  inline hashtags.
- `hashtags` — 5–10 entries, each starts with `#`, no whitespace, no
  case-insensitive duplicates.
- `hooks` — 2–3 opening-line rewrites, each ≤ 140 characters.

All bounds are enforced both by the JSON response schema and by pydantic
validators in `clippos.models.package` — the CLI re-parses the response and
fails with a non-zero exit rather than writing a half-baked pack.

## Producer / consumer

- **Producer**: `scripts/clippos_skill.py package-prompt <workspace>` loads
  `review-manifest.json` + `scoring-request.json`, filters to approved
  candidates, and writes `package-request.json`.
- **Consumer**: `scripts/clippos_skill.py package-save <workspace>` validates
  `package-response.json`, fans per-clip `package.json` files into
  `renders/<clip_id>/`, and emits `package-report.json`.

Both subcommands return structured JSON on stdout so the skill layer can thread
paths straight into its closing response.
