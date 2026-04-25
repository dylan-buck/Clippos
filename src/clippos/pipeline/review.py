from __future__ import annotations

from pathlib import Path

from clippos.models.candidate import CandidateClip
from clippos.models.review import ReviewManifest


def build_review_manifest(
    job_id: str,
    video_path: Path,
    candidates: list[CandidateClip],
    model_scores: list[dict],
) -> ReviewManifest:
    lookup = {
        clip_id: item
        for item in model_scores
        if (clip_id := item.get("clip_id")) is not None
    }
    enriched: list[CandidateClip] = []

    for candidate in candidates:
        extra = lookup.get(candidate.clip_id, {})
        final_score = extra.get("final_score")
        enriched.append(
            candidate.model_copy(
                update={
                    "title": candidate.title
                    if extra.get("title") is None
                    else extra["title"],
                    "hook": candidate.hook
                    if extra.get("hook") is None
                    else extra["hook"],
                    "reasons": candidate.reasons
                    if extra.get("reasons") is None
                    else extra["reasons"],
                    "score": candidate.score if final_score is None else final_score,
                }
            )
        )

    enriched.sort(key=lambda clip: (-clip.score, clip.clip_id))

    return ReviewManifest(job_id=job_id, video_path=video_path, candidates=enriched)
