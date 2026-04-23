# Job Spec

`ClipperJob` is the stable handoff contract for ingest, analysis, review, and render
stages. It defines the source video, destination directory, whether human review is
required before rendering, and the default output profile for downstream exports.
All contract models reject unexpected fields to keep handoffs explicit and stable.

## Contract Summary

- `video_path`: absolute or workspace-resolved source media path
- `output_dir`: destination directory for manifests and rendered clips
- `review_required`: gate that defaults to `true`
- `output_profile.ratios`: default output aspect ratios in stable priority order
- `output_profile.caption_preset`: downstream caption style preset
- `max_candidates`: maximum number of clips surfaced for review

## Validation Rules

- Unknown fields are rejected across the shared contract models
- `max_candidates` must be greater than `0`
- `MediaProbe` numeric fields must all be greater than `0`
- `CandidateClip.start_seconds` and `CandidateClip.end_seconds` must be non-negative
- `CandidateClip.end_seconds` must be greater than `start_seconds`
- `CandidateClip.score` is bounded to `0.0` through `1.0`
- Aspect ratios are restricted to the shared vocabulary: `9:16`, `1:1`, `16:9`

## Example

```json
{
  "video_path": "/tmp/input.mp4",
  "output_dir": "/tmp/out",
  "review_required": true,
  "output_profile": {
    "ratios": ["9:16", "1:1", "16:9"],
    "caption_preset": "hook-default"
  },
  "max_candidates": 12
}
```

## Related Shared Models

- `MediaProbe`: normalized probe metadata used by analysis and render steps
- `CandidateClip`: scored clip candidate with time bounds and rationale
- `ReviewManifest`: review payload tying a job and source video to candidates
- `RenderManifest`: approved clip output map keyed by the shared aspect-ratio vocabulary
