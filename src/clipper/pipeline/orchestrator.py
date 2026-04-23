from __future__ import annotations

from pathlib import Path
from typing import Literal

from clipper.adapters.ffmpeg import probe_media
from clipper.adapters.storage import write_json
from clipper.models.candidate import CandidateClip
from clipper.models.job import ClipperJob
from clipper.models.scoring import ClipBrief
from clipper.pipeline.candidates import mine_windows, to_candidate_clip
from clipper.pipeline.ingest import IngestResult, ingest_job
from clipper.pipeline.review import build_review_manifest
from clipper.pipeline.scoring import (
    ScoringResponseError,
    build_clip_brief,
    build_scoring_request,
    load_scoring_request,
    resolve_scores,
    scores_to_model_payload,
    write_scoring_request,
)
from clipper.pipeline.transcribe import build_transcript_timeline, run_transcription
from clipper.pipeline.vision import build_vision_timeline, run_vision

Stage = Literal["mine", "review", "auto"]
VALID_STAGES: tuple[Stage, ...] = ("mine", "review", "auto")

REVIEW_MANIFEST_FILENAME = "review-manifest.json"


def _canonical_video_path(video_path: Path) -> Path:
    return video_path.expanduser().resolve(strict=False)


def probe_video(video_path: Path) -> dict:
    return probe_media(video_path)


def transcribe_video(video_path: Path, workspace_dir: Path) -> dict:
    return run_transcription(video_path, workspace_dir)


def analyze_video(video_path: Path, workspace_dir: Path) -> dict:
    return run_vision(video_path, workspace_dir)


def score_shortlist(workspace_dir: Path) -> list[dict] | None:
    scores = resolve_scores(workspace_dir)
    if scores is None:
        return None
    return scores_to_model_payload(scores)


def run_job(job: ClipperJob, *, stage: Stage = "auto") -> Path:
    if stage not in VALID_STAGES:
        raise ValueError(f"Unknown stage {stage!r}")

    canonical_video_path = _canonical_video_path(job.video_path)
    resolved_job = job.model_copy(update={"video_path": canonical_video_path})

    probe_data = probe_video(canonical_video_path)
    ingest = ingest_job(resolved_job, probe_data=probe_data)
    workspace_dir = ingest.workspace_dir

    if stage == "review":
        return _finalize_review_stage(ingest, canonical_video_path, workspace_dir)

    request_path = _run_mine_stage(
        resolved_job, ingest, canonical_video_path, workspace_dir
    )
    if stage == "mine":
        return request_path

    model_scores = score_shortlist(workspace_dir)
    if model_scores is None:
        return request_path

    return _write_review_manifest(
        ingest=ingest,
        video_path=canonical_video_path,
        workspace_dir=workspace_dir,
        model_scores=model_scores,
    )


def _run_mine_stage(
    resolved_job: ClipperJob,
    ingest: IngestResult,
    video_path: Path,
    workspace_dir: Path,
) -> Path:
    transcript = build_transcript_timeline(transcribe_video(video_path, workspace_dir))
    vision = build_vision_timeline(analyze_video(video_path, workspace_dir))
    windows = mine_windows(
        transcript, vision, max_candidates=resolved_job.max_candidates
    )
    candidates = [
        to_candidate_clip(rank=index, scored=window)
        for index, window in enumerate(windows)
    ]
    briefs = [
        build_clip_brief(candidate=candidate, scored=window)
        for candidate, window in zip(candidates, windows, strict=True)
    ]
    request = build_scoring_request(
        job_id=ingest.job_id,
        video_path=video_path,
        briefs=briefs,
    )
    return write_scoring_request(workspace_dir, request)


def _finalize_review_stage(
    ingest: IngestResult, video_path: Path, workspace_dir: Path
) -> Path:
    model_scores = score_shortlist(workspace_dir)
    if model_scores is None:
        raise ScoringResponseError(
            "stage=review requires scoring-request.json plus either "
            "scoring-response.json or cached scores for every clip"
        )
    return _write_review_manifest(
        ingest=ingest,
        video_path=video_path,
        workspace_dir=workspace_dir,
        model_scores=model_scores,
    )


def _write_review_manifest(
    *,
    ingest: IngestResult,
    video_path: Path,
    workspace_dir: Path,
    model_scores: list[dict],
) -> Path:
    request = load_scoring_request(workspace_dir)
    if request is None:
        raise ScoringResponseError(
            "Cannot build review manifest without scoring-request.json"
        )
    candidates = [_brief_to_candidate(brief) for brief in request.clips]
    manifest = build_review_manifest(
        ingest.job_id,
        video_path,
        candidates,
        model_scores=model_scores,
    )
    output = workspace_dir / REVIEW_MANIFEST_FILENAME
    write_json(output, manifest.model_dump(mode="json"))
    return output


def _brief_to_candidate(brief: ClipBrief) -> CandidateClip:
    return CandidateClip(
        clip_id=brief.clip_id,
        start_seconds=brief.start_seconds,
        end_seconds=brief.end_seconds,
        score=brief.mining_score,
        reasons=list(brief.mining_signals.reasons),
        spike_categories=list(brief.mining_signals.spike_categories),
    )
