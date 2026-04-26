---
name: clip
description: Local video clipping automation for /clip requests, attached video files, video links, social clip generation, captioned shorts, crop/framing, and rendered 9:16, 1:1, or 16:9 exports. Use this skill whenever the user wants an agent to turn a video into high-potential clips, score clips with the current harness model, approve selected clips, and render final MP4 outputs.
---

# Clip Skill

Use this skill to run the local clipping engine end-to-end from a video path,
attached video file, or video link. The engine does deterministic media work
locally; the current agent supplies the semantic scoring step by reading
`scoring-request.json` and writing `scoring-response.json`.

## Inputs

Accept:

- A local video path, including a path resolved from an attached video file.
- A direct `http` / `https` video URL.
- A platform video URL when `yt-dlp` is installed.
- A messaging-platform attachment URL (Discord CDN, Telegram `api.telegram.org`
  bot-file URL, WhatsApp/Signal-provided HTTPS URL). On Discord and Telegram,
  when the user drops a video into the channel, the attachment appears in the
  message as `{filename, url, size}` — pass that `url` straight into `advance
  --source`. The helper detects CDN hosts and downloads them directly via
  `urllib` (yt-dlp is skipped for those, since they are signed direct mp4s).

Optional user intent:

- Ratios: default all three (`9:16`, `1:1`, `16:9`). Respect explicit requests
  such as "vertical only", "square", "wide", or `--ratios 9:16,1:1`.
- Clip count: default to config `CLIPPOS_APPROVE_TOP` or 5. Treat 5 as the
  normal minimum when the video has enough valid candidate windows.
- Quality threshold: default to config `CLIPPOS_MIN_SCORE` or 0.70.
- Output directory: default to config `CLIPPOS_OUTPUT_DIR` or
  `~/Documents/Clippos`.

If no video source is present, ask one short question for the video link or
file path.

## Slash-command shape

This skill is harness-agnostic. The surface differs per harness but the
workflow below is the same.

**Hermes** exposes a single `/clip` command and treats extra text as a
subcommand or source argument:

- `/clip <video path|url>` — run the main clipping workflow.
- `/clip config [options]` — check or write local defaults.
- `/clip package [workspace]` — generate publish packs after rendering.
- `/clip status` — run the preflight config check.

**Claude Code / Codex** expose three slash commands via `commands/*.md` shims:

- `/clip <video path|url>` — same main workflow.
- `/clip-config [options]` — same as `/clip config`.
- `/clip-package [workspace]` — same as `/clip package`.

When this SKILL.md says "use `/clip config`" or "use `/clip package`", Claude
Code / Codex users substitute `/clip-config` or `/clip-package`. The
underlying helper scripts are identical across harnesses.

## Creator Profile (harness memory)

The skill is memory-aware. Whenever the harness has persistent memory
(Hermes memory, Claude Code `CLAUDE.md` + `~/.claude` memories, Codex
equivalents), check it for creator-profile facts and apply them at the
scoring and packaging handoffs. Creator-profile facts are the kind a content
creator would tell the agent once and expect respected every run:

- Target platform and format (e.g. "TikTok-first, 9:16, 15–45s clips").
- Clip style (e.g. "story-beat > one-liner", "never pick intro music").
- Caption style (e.g. "clean, punchy, no fake hype, no all-caps").
- Brand tone (e.g. "indie-founder, self-deprecating, technical specifics").
- Banned phrases, emoji, or hashtags.
- Title/hook formatting preferences.

Treat these as contextual lens, not rubric overrides: the embedded
`rubric_prompt` and `response_schema` remain authoritative. When no creator
profile is in memory, score and package from the rubric alone and, when the
run finishes, offer to save the user's stated preferences to the harness's
memory tool for next time — do not write directly to `~/.hermes/memories/` or
other memory stores.

## Feedback Loop (self-improving profile)

The skill learns from the user's own keep/skip choices. After every render,
`advance` emits a `feedback_prompt` with the clip IDs. Ask the user which
clips they actually posted (or plan to post) and record the answer:

```bash
"$CLIPPOS_PYTHON" "$CLIPPOS_ROOT/scripts/hermes_clip.py" feedback \
  "$WORKSPACE" --kept c1,c4 --skipped c2,c3 --note c2='too long'
```

Or, for structured payloads from the harness, use `--json` and pipe
`{"entries": [{"clip_id": "c1", "posted": true, "notes": "..."}, ...]}` on
stdin.

Each feedback call writes `feedback-log.json` in the workspace and appends
rows to the global `~/.config/clippos/history.jsonl`. On the next clip
run, `advance` attaches a `creator_patterns` section to the `score` and
`package` handoff payloads. That section contains:

- `summary` — total clips, keep rate, per-ratio and per-spike rates.
- `patterns` — detected regularities, each with a `rule` (human-readable),
  `confidence` (`low` / `medium` / `high`), `sample_size`, and a
  `suggested_memory` string.

Apply the patterns alongside `rubric_prompt` when scoring and packaging.
High-confidence patterns (e.g. "user skips clips over 60s, 92% skip rate
over 18 samples") should strongly bias rubric weights; low-confidence ones
are informational. When a pattern is high confidence and not already in the
harness memory, offer to save its `suggested_memory` via the harness's
memory tool so the rule becomes a stable preference.

Never write directly to `~/.hermes/memories/`. The skill captures outcomes;
the harness owns the memory store.

## Preflight

Run this before the first clip job in a session. The prologue resolves the
skill directory across harnesses:

- **Hermes** substitutes `${HERMES_SKILL_DIR}`.
- **Claude Code / Codex plugins** substitute `${CLAUDE_PLUGIN_ROOT}`.
- **Any other harness** must either pin `CLIPPOS_ROOT` directly or invoke the
  prologue from inside the repo checkout so `$PWD` resolves correctly.

```bash
# Resolve CLIPPOS_ROOT — env var > harness substitution > known install
# locations > persisted config > $PWD. The candidate must contain
# scripts/hermes_clip.py to be accepted.
# Why so many fallbacks: Hermes substitutes HERMES_SKILL_DIR reliably, but
# Claude Code's CLAUDE_PLUGIN_ROOT does not always expand inside command
# bash blocks (Anthropic issue #9354), so we also probe known install
# locations (Claude Code plugin cache, Codex plugin cache, Hermes skill
# dir) and the persisted CLIPPOS_ROOT in the config file.
for candidate in "${CLIPPOS_ROOT:-}" "${HERMES_SKILL_DIR:-}" "${CLAUDE_PLUGIN_ROOT:-}" \
                 "$HOME/.hermes/skills/clip" "$HOME/.claude/skills/clip" \
                 "$HOME/.codex/skills/clip"; do
  if [ -n "$candidate" ] && [ -f "$candidate/scripts/hermes_clip.py" ]; then
    CLIPPOS_ROOT="$candidate"; break
  fi
done
# Claude Code + Codex marketplace installs land under a versioned cache
# (~/.claude/plugins/cache/<marketplace>/clip/<sha>, ~/.codex/plugins/
# cache/<marketplace>/clip/<sha>); pick the newest one if no env var hit.
if [ -z "${CLIPPOS_ROOT:-}" ]; then
  for cache_root in "$HOME/.claude/plugins/cache" "$HOME/.codex/plugins/cache"; do
    [ -d "$cache_root" ] || continue
    candidate="$(find "$cache_root" -mindepth 3 -maxdepth 3 -type d -name "clip" -path "*/clip" 2>/dev/null | head -1)"
    [ -n "$candidate" ] && [ -f "$candidate/scripts/hermes_clip.py" ] && \
      { CLIPPOS_ROOT="$candidate"; break; }
  done
fi
if [ -z "${CLIPPOS_ROOT:-}" ] && [ -f "$HOME/.config/clippos/.env" ]; then
  CLIPPOS_ROOT="$(awk -F= '/^CLIPPOS_ROOT=/{gsub(/^["'"'"']|["'"'"']$/,"",$2); print $2; exit}' "$HOME/.config/clippos/.env")"
fi
[ -n "${CLIPPOS_ROOT:-}" ] && [ -f "$CLIPPOS_ROOT/scripts/hermes_clip.py" ] || \
  { [ -f "$PWD/scripts/hermes_clip.py" ] && CLIPPOS_ROOT="$PWD"; }
# v1.x bootstrap: native plugin managers (Claude Code's /plugin, Codex's
# `codex marketplace add`) clone the repo but do not run pip — there is
# no PostInstall hook for engine extras. The bootstrap script creates a
# .venv at the install root and pip-installs the engine extras (~5 min,
# ~700 MB of wheels). Idempotent — no-op once the .venv exists. Hermes
# users run this script directly post-clone, so they don't pay this
# cost on first /clip.
[ -d "$CLIPPOS_ROOT/.venv" ] || bash "$CLIPPOS_ROOT/scripts/bootstrap-venv.sh"
CLIPPOS_PYTHON="${CLIPPOS_PYTHON:-$CLIPPOS_ROOT/.venv/bin/python}"
[ -x "$CLIPPOS_PYTHON" ] || CLIPPOS_PYTHON="$(command -v python3)"
"$CLIPPOS_PYTHON" "$CLIPPOS_ROOT/scripts/clip_skill.py" config-check
```

If discovery fails (no env var, no install paths, no persisted config,
not in the repo), persist the path once and every future invocation
resolves cleanly:

```bash
"$CLIPPOS_ROOT/.venv/bin/python" "$CLIPPOS_ROOT/scripts/clip_skill.py" \
  config-write --root "$CLIPPOS_ROOT"
```

`bootstrap-venv.sh` runs that step automatically as its last action.

### Diarization (zero-config by default)

The skill ships with a zero-config open-source diarizer (silero-VAD +
SpeechBrain ECAPA-TDNN + spectral clustering). No HuggingFace token, no
license click-through. Models are public and auto-cache on first use.

`CLIPPOS_DIARIZER` (env var or `--diarizer` flag) chooses the path:

- `speechbrain` (default) — open-source, no setup. Recommended.
- `pyannote` — opt-in upgrade. Requires `HF_TOKEN` and one-time license
  acceptance at <https://hf.co/pyannote/speaker-diarization-3.1>. Slightly
  higher quality but the gate is a real onboarding cost.
- `off` — skip diarization entirely. Every segment gets `SPEAKER_00`. Use
  for single-speaker content where speaker labels add no value.

Only mention the pyannote path if the user explicitly asks for higher
diarization quality.

The real pipeline needs FFmpeg, ffprobe, an FFmpeg build with the `ass`
subtitle filter (libass), and engine extras. **HF_TOKEN is no longer
required** for the default open-source diarizer; the preflight reports it
as missing but the run will succeed without it. Only ask the user for an
HF token if they explicitly want the pyannote upgrade (see "Diarization"
above).

Every subsequent bash block assumes `CLIPPOS_ROOT` and `CLIPPOS_PYTHON` are
resolved with the same four-line prologue.

## Main Workflow (agent loop)

Prefer the `scripts/hermes_clip.py` driver (named for its Hermes-first design,
but harness-agnostic — it works anywhere a Python script can shell out + read
JSON). It advances the pipeline state machine and always prints a single JSON
payload describing the next action, so each `/clip` turn is exactly one tool
call plus (when needed) one model handoff.

1. Start or resume a job:

```bash
"$CLIPPOS_PYTHON" "$CLIPPOS_ROOT/scripts/hermes_clip.py" advance --source "$SOURCE"
```

Use `--ratios 9:16,1:1` or `--clips 2` only when the user asks. The payload
contains `workspace`, `next_action`, and—when a model handoff is required—
`handoff_request_path` and `handoff_response_path`.

2. When `next_action == "brief"` (v1.1):

- Read the request at `handoff_request_path` (`brief-request.json`).
- Follow its embedded `brief_prompt` and `response_schema`.
- Read the `transcript_excerpt`. The full transcript is provided when
  short; for long videos, `transcript_truncated: true` and the excerpt
  contains the head + tail with a marker where the middle was dropped.
  Infer the global shape from what you can see plus `speakers` +
  `duration_seconds`.
- Author a tight, opinionated `VideoBrief`: `theme`, `video_format`,
  3–5 `expected_viral_patterns`, 0–3 `anti_patterns`, optional
  `audience` / `tone` / `notes`. The brief must be *scoring-relevant*
  — what to up-weight and down-weight in this specific video — not a
  summary.
- Write `handoff_response_path` (`brief-response.json`) with a valid
  `VideoBriefResponse`.
- Re-run advance on the workspace. The brief is cached for the rest
  of the workspace's lifetime; the next per-clip scoring call sees
  the brief as `video_brief` on the scoring request.

This step is one model call per video and is the highest-leverage
moment in the loop — it is where the agent's context-synthesis
ability outperforms the per-clip rubric. Skip it only when
`next_action != "brief"` (i.e. `output_profile.video_brief: false`
in the job).

3. When `next_action == "score"`:

- Read the request at `handoff_request_path`.
- Follow its embedded `rubric_prompt` and `response_schema`.
- Check loaded harness memory for any **creator profile** preferences before
  scoring — target platform (TikTok/Reels/Shorts), preferred clip length,
  caption style, brand tone, hook style, and banned topics or phrases. Let
  those bias scores, spike categories, and penalties. The rubric stays
  authoritative; creator preferences act as tiebreakers and contextual lens.
- If the advance payload includes a `creator_patterns` field (populated from
  past feedback), treat high-confidence patterns as strong bias signals and
  medium/low as informational. See the "Feedback Loop" section above.
- Score every `ClipBrief`, preserving `clip_id` and `clip_hash` verbatim.
- Write `handoff_response_path` with a valid `ScoringResponse`.
- Re-run advance on the workspace:

```bash
"$CLIPPOS_PYTHON" "$CLIPPOS_ROOT/scripts/hermes_clip.py" advance --workspace "$WORKSPACE"
```

  Advance builds the review manifest, auto-approves the top-scoring clips
  above the configured threshold, fills from the next-best clips when needed
  to reach the requested count, and renders the approved clips.

4. When `next_action == "done-renders"`, the payload includes `clips_dir`,
   `clips[]` with `renders` keyed by ratio, and a `feedback_prompt` with the
   clip IDs. Return the MP4 paths plus the clips directory and workspace path
   to the user, then ask which clips they kept or plan to post. Pipe the
   answer into `hermes_clip.py feedback` (see the "Feedback Loop" section) so
   the creator profile keeps learning. Mention if any requested ratio was
   skipped.

5. When `next_action == "error"`, surface the `error` string and the `stage`
   it happened in. Do not retry silently—diagnose or ask the user.

### Deterministic fallback (raw primitives)

If the harness cannot use `hermes_clip.py`, the older step-by-step flow still
works. Run `prepare` → `clippos.cli run --stage mine` → score → `--stage
review` → `clip_skill.py approve` → `--stage render` → `clip_skill.py outputs`.
See git history for the long form; `hermes_clip.py` encodes the same sequence.

## Packaging Workflow

Use `/clip package` after a render finishes to produce per-clip publish packs
(5+ title candidates, thumbnail overlay lines, a social caption, hashtags, and
opening-line hooks). The `hermes_clip.py` driver handles workspace resolution
and packaging handoff; it uses the newest job workspace when none is given.

1. Resume the latest workspace (or pass one explicitly) and advance through
   packaging:

```bash
WORKSPACE=$("$CLIPPOS_PYTHON" "$CLIPPOS_ROOT/scripts/hermes_clip.py" latest-workspace --plain)
"$CLIPPOS_PYTHON" "$CLIPPOS_ROOT/scripts/hermes_clip.py" advance --workspace "$WORKSPACE" --package
```

2. When `next_action == "package"`:

- Read `handoff_request_path` (`package-request.json`).
- Follow its embedded `package_prompt` and `response_schema`.
- Apply any **creator profile** rules from harness memory: caption style
  (clean vs. punchy vs. verbose), brand tone, banned phrases or emoji,
  hashtag preferences, hook pattern, and title formatting. The schema is
  authoritative; preferences shape phrasing inside it.
- If the advance payload includes `creator_patterns`, let high-confidence
  patterns guide hook length, pacing references, and hashtag vocabulary.
- Produce one `PublishPack` per clip preserving `clip_id` + `clip_hash`.
- Write `handoff_response_path` (`package-response.json`).
- Re-run advance:

```bash
"$CLIPPOS_PYTHON" "$CLIPPOS_ROOT/scripts/hermes_clip.py" advance --workspace "$WORKSPACE"
```

3. When `next_action == "done-package"`, the payload includes `clips[]` with
   `renders` and `package` paths. Report the per-clip `package.json` paths
   alongside the rendered MP4 paths so the user can paste titles, captions,
   and hashtags straight into the upload form.

## Config Workflow

Use `/clip config` when setup is missing or the user wants defaults changed.
The helper stores config at `~/.config/clippos/.env`:

```bash
"$CLIPPOS_PYTHON" "$CLIPPOS_ROOT/scripts/clip_skill.py" config-write \
  --output-dir "$HOME/Documents/Clippos" \
  --ratios "9:16,1:1,16:9" \
  --max-candidates 12 \
  --approve-top 5 \
  --min-score 0.70
```

If the user gives a Hugging Face token, add `--hf-token "$HF_TOKEN"`. Treat the
token as sensitive and do not print it back.

## Output Contract

End the response with:

- Workspace path.
- Approved clip IDs and titles when available.
- Rendered MP4 paths for each ratio.
- Any setup or render failures with the exact command that failed.

Do not call the job production-ready until a real-video `/clip` run has
completed on the user's machine with engine extras and `HF_TOKEN` configured.
