# Miner quality improvements

The candidate miner in `src/clippos/pipeline/candidates.py` is
calibrated for viral-monologue patterns (single-speaker, hook-driven,
high-keyword-density). It systematically under-emits candidates from
multi-speaker interview / podcast / guest-Q&A content, even when that
content is the most clip-worthy material in the source.

This is a **pre-brief-stage engine quality bug**, not a v1.1 feature.
The [v1.1 video brief](v1.1.md) cannot rescue a window the miner
never proposed — so this work has to land *before* the brief is built,
otherwise the brief will silently fail on interview-format content.

---

## Empirical evidence (2026-04-25 dogfood)

Source: 28-minute single-host crypto stream recap that brings on a
guest trader for the back third. Diarization correctly identified:

| Speaker | Speech time | Present |
|---------|-------------|---------|
| SPEAKER_00 (host) | 1128s | 0–28 min |
| SPEAKER_01 (guest) | **304s** | **18:49–26:52** |
| SPEAKER_02 (cameo) | 33s | 16:07–16:41 |

The guest had two substantial blocks:

| Block | Range | Duration | Sample |
|-------|-------|----------|--------|
| Casual intro | 18:49–22:24 | 3.6 min | *"I'm a massive Robinhood holder…"* |
| **Stock picks** | **22:47–26:04** | **3.3 min** | **"Leopold Ashbender playbook, Bloom Energy… one of the best investors of all time"** |

That second block is exactly the kind of expert-endorsement /
specific-pick content that is highly clippable for a finance audience
— and which a human editor would obviously surface.

**Mining produced 0 candidate windows from 1129–1612s.** The entire
8-minute guest segment was invisible to the scoring stage. Coverage
plot:

```
0────────600───────1129───────1612──────1690s
[host]   [host]     [─── GUEST INTERVIEW ───][host]
         ████  ██                              █
         ↑↑↑   ↑↑                              ↑
         5 candidate windows clustered here     nothing
```

This is not a "model judged the guest content low" outcome. The
mining stage simply did not propose those windows for scoring.

---

## Root cause — why the miner can't see interview content

`MiningConfig.weights` (lines 140–152 of `candidates.py`):

```python
hook                = 0.12   # buzzword-driven monologue openers
keyword             = 0.12   # "crazy", "insane", "the truth", "secret"
payoff              = 0.12   # concrete claims after setup
question_to_answer  = 0.08
motion              = 0.08
shot_change         = 0.05
face_presence       = 0.05
speaker_interaction = 0.05   # ← the only multi-speaker signal
delivery_variance   = 0.03
```

For a measured, conversational interview turn:

- `hook` signal: **low** (guests don't open with "let me tell you").
- `keyword` signal: **low** (no monologue buzzwords).
- `payoff` signal: low–medium (concrete picks fire it; setup chat does not).
- `speaker_interaction`: **high** (many transitions).
- `question_to_answer`: high (host asks, guest answers).

The dominant signals (hook + keyword + payoff = **0.36 combined
weight**) are calibrated for solo content. Multi-speaker content has
**a single 0.05-weight signal** to compete with that, plus 0.08 for
Q→A. Even a perfectly-clipped interview turn maxes its multi-speaker
contribution at ~0.13 vs. up to 0.36 for a single hooky monologue
sentence.

Worse, the score floor is `0.35`. So a 0.30-scoring guest-stock-pick
window doesn't even enter `min_candidates` backfill consideration — it
gets dropped before backfill runs, because backfill only operates on
already-emitted-but-below-floor windows. (Actually it does enter
backfill consideration — *if* it was emitted as a window in the first
place. The dedup pass with `max_overlap_ratio=0.5` may collapse
several similar guest windows into one then drop it for being below
the keyword/hook signal even before scoring. This is worth tracing
during implementation.)

`derive_spike_categories` (lines 450–470) compounds the problem.
`emotional_confrontation` requires `keyword >= 0.16 AND
speaker_interaction >= 0.1` — keyword gating means even strong
multi-speaker content without buzzwords gets no spike upgrade.

---

## Fix proposals (in priority order)

### M1. Reweight `speaker_interaction` to monologue-equivalent

Bump from `0.05` → `0.12` (parity with hook/keyword/payoff). One-line
change in `ScoringWeights`. Run the existing unit tests to confirm
monologue content doesn't regress catastrophically.

**Cost:** trivial. **Risk:** monologue scores shift slightly; the
existing `min_candidates=5` floor protects against under-selection.
**Coverage:** cheapest possible improvement; should be the first thing
tried.

### M2. Add explicit interview-block emission

Detect contiguous multi-speaker stretches (>30s of alternation) at the
windowing stage. For each stretch, explicitly emit windows centered on
speaker transitions, regardless of monologue-keyword scores. These
windows enter scoring on their merits and the model decides if they
are clip-worthy.

**Implementation:** new pass in the candidate emission loop that runs
*alongside* the existing keyword-driven emission. Marks emitted windows
with `source="speaker_transition"` for downstream debugging.

**Cost:** medium (new emission path, dedup interaction needs care).
**Risk:** more candidates → higher token cost in scoring. Bounded by
the existing `max_candidates` cap.
**Coverage:** directly fixes the dogfood failure mode.

### M3. Conversational keyword bucket

The existing `KEYWORD_PHRASES` (search `candidates.py`) is tuned for
viral monologue: *crazy, insane, the truth, secret, let me tell you*.
Add a parallel `INTERVIEW_KEYWORD_PHRASES` bucket tuned for
interview-of-expert content:

- "the play here is"
- "what's your take on"
- "I think the most important thing"
- "if you want to follow"
- "hands down one of the best"
- "the way I think about it"
- "my biggest position is"
- "I'm long" / "I'm short"
- "the trade I like"

Activated when `speaker_interaction >= 0.1`. Otherwise dormant so it
doesn't pollute monologue scoring.

**Cost:** small (data-only). **Risk:** keyword tuning is empirical;
needs iteration on a few real podcasts.
**Coverage:** complements M1+M2; pushes scoring toward interview-style
payoff phrases the rubric currently misses.

### M4. Add `expert_endorsement` and `specific_pick` spike categories

Currently `derive_spike_categories` emits: `controversy`,
`emotional_confrontation`, `absurdity`, `action`,
`unusually_useful_claim`. None capture the "guest expert says specific
thing" pattern that's clippable in finance/business content.

Two new categories:

- **`expert_endorsement`** — fires when `speaker_interaction >= 0.1`
  AND the window contains attribution language ("hands down", "the
  best", "one of the most", followed by a specific reference).
- **`specific_pick`** — fires when `numeric >= 0.1` AND
  `speaker_interaction >= 0.1` AND the window contains ticker-symbol
  patterns or stock-name patterns.

Bumps `RUBRIC_VERSION` (since spike categories are part of the
scoring contract). Same precedent as `e0ab071` adding `big_number`.

**Cost:** medium (new categories, rubric bump, scoring prompt update).
**Risk:** false positives on casual mentions. Should require at least
two pattern matches per window.
**Coverage:** finance/podcast vertical specifically.

### M5. Lower the score floor inside detected multi-speaker stretches

A 0.30 score on a guest stock pick *should* be a candidate; the same
0.30 score on a monologue ramble probably should not. Make the floor
context-dependent:

```python
score_floor: float = 0.35           # monologue
multi_speaker_score_floor: float = 0.20  # inside detected interview block
```

Apply during the candidate selection pass (after scoring, before
dedup). Combined with M1–M4 the threshold gap should shrink — but a
context-dependent floor is the right safety net for content the
heuristics still under-weight.

**Cost:** small. **Risk:** more candidates from interview blocks; some
may be uninteresting. The model rejects those during scoring.
**Coverage:** safety net for cases M1–M4 don't fully cover.

---

## Test plan

1. **Regression fixture** — extract the 1129–1612s window from the
   2026-04-25 dogfood transcript + diarization (anonymize speaker
   text if needed) and add it to `tests/pipeline/test_candidates.py`
   as a permanent fixture. Assert that mining emits at least one
   candidate covering the Bloom Energy mention (around minute 22:47).
2. **Existing fixtures** — re-run all monologue-style candidate tests
   to confirm M1 (weight bump) doesn't drop monologue selections
   below `min_candidates`.
3. **Synthetic interview** — build a tiny synthetic transcript with
   two speakers alternating Q→A around a numeric claim, and assert
   M2 (explicit interview emission) surfaces it.

## Validation gate

Re-run `/clippos` on the same crypto stream after M1–M3 land. Inspect
`scoring-request.json`:

- **MUST**: at least one candidate window covers 22:47–26:04 (the
  Bloom Energy block).
- **MUST**: the previously-missed Robinhood/setup chat block (18:49–
  22:24) is at least proposed as a candidate, even if the model
  later scores it low.
- **SHOULD**: the existing 5 monologue candidates still appear, with
  at most a 0.05 score drop (acceptable cost of weight rebalancing).

If the Bloom Energy window scores >0.45 once the v1.1 brief is also
in place, the brief + miner combination is working as intended.

---

## Sequencing

1. **Land pre-ship-fixes first** (see [pre-ship-fixes.md](pre-ship-fixes.md)).
   Without them, the install path is broken for new users and we
   cannot dogfood iteratively.
2. **M1 (one-line weight bump) immediately after.** Cheapest possible
   intervention; may move the needle alone.
3. **Re-run the same dogfood** with M1 only. If guest content now
   surfaces, M2–M5 may be unnecessary or smaller in scope.
4. **M2 + M3 + M4 as a single batch** if M1 is insufficient. They
   compose; testing them together amortizes the dogfood cost.
5. **M5 as a final safety net** after M2–M4 telemetry shows whether
   it's still needed.
6. **Then v1.1 brief**, on top of the now-richer candidate set.
