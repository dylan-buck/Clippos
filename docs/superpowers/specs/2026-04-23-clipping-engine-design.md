# Clipping Engine V1 Design

Date: 2026-04-23
Topic: Local-first clipping engine for Claude Code, Codex, and Hermes Agent
Status: Draft approved in conversation, written for review

## Goal

Build a shared local clipping engine with thin wrappers for Claude Code, Codex, and Hermes Agent that:

- analyzes the entire video
- finds clips with high view potential
- applies hook-oriented caption styling
- renders approved clips in 9:16, 1:1, and 16:9

V1 optimizes for highest practical quality over lowest cost or simplest implementation.

## Product Boundary

V1 is a shared engine with thin agent-specific wrappers.

- The shared engine owns media analysis, candidate generation, scoring, review artifacts, caption planning, reframing, and rendering.
- The wrappers only collect inputs, start jobs, surface review results, and trigger renders.
- V1 is local-only.
- V1 includes a required human review step before final renders.

This keeps quality logic centralized and avoids maintaining three separate implementations.

## Target Content

Primary target:

- long-form talking head footage
- podcasts
- recorded livestreams
- vlogs with a dominant speaker

The engine should prefer speech-heavy content with enough visual and audio variation to support clip ranking, caption emphasis, and reframing.

## Design Principles

- Quality-first over cheapest-first
- Balanced scoring over purely retention-first or purely virality-first
- Deterministic media analysis before LLM judgment
- Cached intermediates so reruns are fast
- Human approval before export
- Machine-readable outputs so future automation is easier

## High-Level Architecture

### 1. Wrapper Layer

Claude Code, Codex, and Hermes each expose a thin skill or automation that:

- accepts a local video path and job options
- invokes the shared engine
- displays the review package
- lets the user approve clips for export

The wrapper should not own scoring or rendering logic.

### 2. Core Engine

The shared engine runs a fixed pipeline:

1. Ingest
2. Extract
3. Analyze
4. Candidate mining
5. Harness-model review
6. Human approval
7. Final render

### 3. Output Layer

The engine writes:

- cached intermediate analysis artifacts
- review package artifacts
- approved multi-ratio renders
- caption assets
- machine-readable manifests

## Pipeline

### 1. Ingest

Input:

- local video file
- job profile
- optional caption/style preset

Actions:

- probe media with FFmpeg
- normalize metadata
- establish cache keys
- prepare output workspace

### 2. Extract

Actions:

- extract audio
- sample frames or low-rate visual proxies
- align timeline metadata
- persist reusable artifacts for later reruns

### 3. Analyze

Actions:

- transcription
- speaker diarization
- shot or visual change detection
- face presence and prominence tracking
- motion and framing analysis
- audio emphasis and silence analysis

Outputs:

- word- or phrase-aligned transcript
- speaker segments
- visual event timeline
- audio event timeline
- merged signal timeline

### 4. Candidate Mining

The engine scans the full video to create candidate windows before any semantic model scoring.

Signals include:

- transcript spikes such as surprise, stakes, numbers, conflict, taboo, strong claims, and unusual phrasing
- delivery spikes such as raised intensity, interruptions, pace changes, emphasis, or abrupt pauses
- structure cues such as question-to-answer, setup-to-payoff, reveal moments, or topic shifts
- visual cues such as motion jumps, shot changes, stronger face presence, or visible reactions

The miner proposes overlapping windows, merges related windows, and normalizes them into clip candidates.

### 5. Harness-Model Review

The harness model evaluates only shortlisted candidate clips.

It scores:

- hook strength in the first 1-3 seconds
- shareability
- standalone clarity
- payoff strength
- delivery energy
- quotability

It also applies explicit positive weighting to spike categories:

- controversy
- taboo
- absurdity
- action or intensity
- emotional confrontation
- unusually useful claim

Those categories must be multimodal:

- semantic evidence from transcript content
- performance evidence from pacing, overlap, intensity, or pauses
- visual evidence from motion, facial reaction, framing change, or on-screen action

The harness-model pass must also apply strong penalties for:

- buried lead
- rambling middle
- missing payoff
- confusing setup requirements
- clips that only work with prior context

### 6. Human Review

The user reviews top-ranked clips before any render job starts.

The review step should allow:

- accept or reject a clip
- adjust start and end times
- inspect reasons for selection
- inspect transcript excerpt and standout line
- inspect recommended aspect-ratio priority

### 7. Final Render

After approval, the engine renders all three aspect ratios by default:

- 9:16
- 1:1
- 16:9

Render includes:

- crop and framing plan per ratio
- caption burn-in with hook-forward styling
- safe margins and per-ratio layout adjustments
- final MP4 export
- sidecar caption and manifest outputs

## Scoring Policy

V1 uses a balanced policy.

This means:

- the engine should reward spike potential and shareability
- the engine should not become a pure shock or controversy detector
- a chaotic or confrontational moment can win, but only if it still works as a coherent clip

Operationally, the score should combine:

- semantic meaning
- delivery and vocal performance
- visual performance

Balanced scoring is intended to catch clips that are viral because they are provocative, silly, taboo, intense, or emotionally sharp, while still making sense to a new viewer.

## Tooling Recommendations

### Core stack

- Python-first core engine
- FFmpeg for media probing, extraction, cutting, scaling, and export
- OpenCV and lightweight tracking for motion, face presence, and reframing anchors

### Speech and alignment

- Whisper-family transcription
- Prefer `WhisperX` or a `faster-whisper` pipeline with alignment
- Separate diarization module, preferably `pyannote` if setup cost is acceptable

### Semantic scoring

- Harness model only for shortlisted candidate evaluation, explanations, clip titles, hook copy, and caption-style decisions
- Do not use the harness model to process the full raw video end to end

### Agent integration

- Thin wrappers for Claude Code, Codex, and Hermes should call the shared engine using a common job spec

## Outputs

### Review package

The engine should generate a review package before rendering.

Per clip, include:

- timestamps
- confidence or ranking score
- explanation for why the clip was selected
- spike category labels
- suggested title
- suggested opening hook
- transcript snippet
- standout quoted line
- recommended aspect-ratio priority

### Final render outputs

For each approved clip:

- captioned MP4 in 9:16
- captioned MP4 in 1:1
- captioned MP4 in 16:9
- sidecar SRT or ASS
- sidecar JSON manifest with clip metadata and rendering decisions

## Caption System

V1 should support hook-oriented captions rather than plain subtitles.

The caption system should:

- emphasize key words or phrases
- preserve readability across all three ratios
- maintain safe margins
- support style presets later

Brand-specific styling can wait until after the base engine is stable.

## Reframing System

V1 reframing should be driven by:

- active speaker region
- face prominence
- motion center
- composition fallback rules

The goal is robust framing for speech-heavy footage, not perfect cinematic reframing for every content type.

## Non-Goals For V1

- full cloud or hosted execution
- autonomous publishing without human review
- perfect support for every video genre
- deeply brand-custom caption design
- production claims based on benchmark data that does not yet exist

## Risks

- False positives from spike-heavy but context-dependent moments
- Weak reframing on visually messy or multi-subject footage
- Caption styling that feels generic before style presets exist
- Evaluation quality limited until accepted versus rejected clip data is collected

## Why This Design

This design matches the stated constraints:

- highest quality matters more than lowest complexity
- local-only is preferred
- code efficiency matters, so full-video model usage should be minimized
- clip quality should come from a real signal pipeline, not prompt-only guessing
- all three agents should share one engine

The key strategic choice is deterministic full-video mining first, then semantic harness-model review on a shortlist. That keeps the expensive reasoning where it matters and makes the system easier to debug and improve.

## Expected Next Step

After spec approval, the next step is an implementation plan for:

- repository structure
- job spec format
- analysis pipeline modules
- scoring contract
- review artifact format
- render pipeline
- wrapper interfaces for Claude Code, Codex, and Hermes
