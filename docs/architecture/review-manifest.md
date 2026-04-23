# Review Manifest

`ReviewManifest` is the review-stage package passed from candidate generation into human or model-assisted review.

## Fields

- `job_id`: stable job identifier for the source clipper run.
- `video_path`: source video file used to derive candidate clips.
- `candidates`: ordered list of `CandidateClip` records prepared for review.

## Candidate enrichment

The review pipeline builds the manifest with `build_review_manifest(...)`. It accepts the raw candidate list plus the harness scoring output normalized by `scores_to_model_payload(...)`:

```python
{
    "clip_id": "clip-000",
    "title": "He admits the hidden tradeoff",
    "hook": "Nobody tells you this part",
    "reasons": ["strong hook", "clear payoff"],
    "final_score": 0.88,
}
```

Candidates are enriched by `clip_id`:

- `title`, `hook`, and `reasons` fall back to the original candidate values when the harness does not return an override.
- `score` is replaced by `final_score` when present, so the manifest ordering reflects the harness judgment rather than the mining score.
- Unmatched model score entries are ignored.
- Enriched candidates are sorted by `(-score, clip_id)` so the strongest clips surface first with a deterministic tiebreaker.

## Scoring source

The harness-side input is produced during the scoring handoff. `run_job(..., stage="auto")` and `stage="review"` both call `score_shortlist(workspace_dir)`, which merges `scoring-response.json` and the per-clip cache into a list of `ClipScore` records before flattening to the payload shape above. See [scoring-handoff.md](scoring-handoff.md) for the full pipeline.

## Persistence

`run_job(...)` currently persists this package to `<output_dir>/jobs/<job_id>/review-manifest.json` with `clipper.adapters.storage.write_json(...)`, typically using `manifest.model_dump(mode="json")` as the payload.
