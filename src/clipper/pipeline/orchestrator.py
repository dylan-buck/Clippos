from __future__ import annotations

from pathlib import Path
from typing import Literal

from clipper.adapters import ffmpeg_render
from clipper.adapters.ffmpeg import probe_media
from clipper.adapters.storage import read_json, write_json
from clipper.models.analysis import MediaProbe
from clipper.models.candidate import CandidateClip
from clipper.models.job import ClipperJob
from clipper.models.render import RenderManifest
from clipper.models.review import ReviewManifest
from clipper.models.scoring import ClipBrief, VideoBrief
from clipper.pipeline.brief import (
    BriefResponseError,
    brief_request_path,
    build_brief_request,
    load_brief_request,
    resolve_brief,
    write_brief_request,
)
from clipper.pipeline.candidates import mine_windows, to_candidate_clip
from clipper.pipeline.ingest import IngestResult, ingest_job
from clipper.pipeline.render import (
    build_render_plan,
    clip_render_dir,
    render_manifest_path,
)
from clipper.pipeline.review import build_review_manifest
from clipper.pipeline.scoring import (
    ScoringResponseError,
    build_clip_brief,
    build_scoring_request,
    clip_brief_with_recomputed_hash,
    compute_video_brief_hash,
    load_scoring_request,
    resolve_scores,
    scores_to_model_payload,
    write_scoring_request,
)
from clipper.pipeline.transcribe import build_transcript_timeline, run_transcription
from clipper.pipeline.vision import build_vision_timeline, run_vision

Stage = Literal["mine", "brief", "review", "render", "auto"]
VALID_STAGES: tuple[Stage, ...] = ("mine", "brief", "review", "render", "auto")

REVIEW_MANIFEST_FILENAME = "review-manifest.json"
RENDER_REPORT_FILENAME = "render-report.json"


class RenderStageError(RuntimeError):
    pass


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


def render_clip(manifest: RenderManifest) -> list[ffmpeg_render.RenderResult]:
    return ffmpeg_render.render_clip(manifest)


def run_job(job: ClipperJob, *, stage: Stage = "auto") -> Path:
    if stage not in VALID_STAGES:
        raise ValueError(f"Unknown stage {stage!r}")

    canonical_video_path = _canonical_video_path(job.video_path)
    resolved_job = job.model_copy(update={"video_path": canonical_video_path})

    probe_data = probe_video(canonical_video_path)
    ingest = ingest_job(resolved_job, probe_data=probe_data)
    workspace_dir = ingest.workspace_dir

    if stage == "render":
        return _finalize_render_stage(
            job=resolved_job,
            ingest=ingest,
            video_path=canonical_video_path,
            workspace_dir=workspace_dir,
        )

    if stage == "review":
        return _finalize_review_stage(ingest, canonical_video_path, workspace_dir)

    if stage == "brief":
        # Standalone brief stage: rebuilds scoring-request.json with the
        # brief embedded, assuming brief-response.json has been written
        # by the harness. Useful for raw-CLI users who want to drive the
        # brief handoff without going through `auto`.
        return _finalize_brief_stage(
            ingest=ingest,
            video_path=canonical_video_path,
            workspace_dir=workspace_dir,
        )

    request_path = _run_mine_stage(
        resolved_job, ingest, canonical_video_path, workspace_dir
    )
    if stage == "mine":
        return request_path

    # v1.1 brief stage (gated on output_profile.video_brief). Auto pauses
    # here when brief is enabled and the harness has not yet authored
    # brief-response.json — same pattern as the scoring handoff below.
    if resolved_job.output_profile.video_brief:
        try:
            brief = resolve_brief(workspace_dir)
        except BriefResponseError:
            # Surface invalid brief responses; the caller (CLI / hermes)
            # decides how to present this. Returning the brief request
            # path here would mask the real error.
            raise
        if brief is None:
            # Waiting on harness brief authoring. Return the brief
            # request path so the caller knows where to look.
            return brief_request_path(workspace_dir)
        # Brief resolved — embed it into scoring-request.json so the
        # harness scoring step sees it. Idempotent: re-runs with the
        # same brief produce the same scoring-request.json content.
        _rewrite_scoring_request_with_brief(
            workspace_dir=workspace_dir,
            brief=brief,
        )

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
    request_path = write_scoring_request(workspace_dir, request)
    # v1.1: also write brief-request.json so the harness has the prompt
    # ready when video_brief is enabled. We always write the request so
    # the harness can author a brief even if the orchestrator is later
    # invoked with video_brief=False; the consumer chooses whether to
    # use the result.
    if resolved_job.output_profile.video_brief:
        brief_request = build_brief_request(
            job_id=ingest.job_id,
            video_path=video_path,
            transcript_timeline=transcript,
        )
        write_brief_request(workspace_dir, brief_request)
    return request_path


def _finalize_brief_stage(
    *,
    ingest: IngestResult,
    video_path: Path,
    workspace_dir: Path,
) -> Path:
    """Re-write scoring-request.json with the resolved brief embedded.

    Assumes mine has run (scoring-request.json exists) and the harness
    has produced a valid brief-response.json. Returns the path to the
    rewritten scoring-request.json. Used by ``--stage brief`` CLI
    invocations and by the auto stage path when the brief becomes
    available.
    """
    if load_brief_request(workspace_dir) is None:
        raise BriefResponseError(
            "stage=brief requires brief-request.json; run stage=mine first "
            "with output_profile.video_brief=True"
        )
    try:
        brief = resolve_brief(workspace_dir)
    except BriefResponseError:
        raise
    if brief is None:
        raise BriefResponseError(
            "stage=brief requires brief-response.json (or a cached brief "
            "matching the current rubric_version + job_id)"
        )
    return _rewrite_scoring_request_with_brief(
        workspace_dir=workspace_dir,
        brief=brief,
    )


def _rewrite_scoring_request_with_brief(
    *,
    workspace_dir: Path,
    brief: VideoBrief,
) -> Path:
    """Update scoring-request.json so it carries the resolved video_brief.

    Idempotent: if the existing request already embeds the same brief,
    we still rewrite (cheap; preserves canonical formatting). The clip
    The rubric_prompt is preserved verbatim. Clip hashes are re-derived with
    the brief digest because the brief is now part of the scoring context.
    """
    request = load_scoring_request(workspace_dir)
    if request is None:
        raise ScoringResponseError(
            "Cannot embed brief into scoring-request.json — file is missing "
            "or invalid; run stage=mine first"
        )
    scoring_context_hash = compute_video_brief_hash(brief)
    clips = [
        clip_brief_with_recomputed_hash(
            clip,
            scoring_context_hash=scoring_context_hash,
            rubric_version=request.rubric_version,
        )
        for clip in request.clips
    ]
    updated = request.model_copy(update={"clips": clips, "video_brief": brief})
    return write_scoring_request(workspace_dir, updated)


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


def _finalize_render_stage(
    *,
    job: ClipperJob,
    ingest: IngestResult,
    video_path: Path,
    workspace_dir: Path,
) -> Path:
    review_manifest = _load_review_manifest(workspace_dir)
    if review_manifest is None:
        raise RenderStageError(
            "stage=render requires review-manifest.json; run stage=review first"
        )
    if not review_manifest.candidates:
        raise RenderStageError("review-manifest.json has no candidates to render")
    approved_candidates = [
        candidate for candidate in review_manifest.candidates if candidate.approved
    ]
    if not approved_candidates:
        raise RenderStageError(
            "review-manifest.json has no approved candidates to render"
        )

    transcript = build_transcript_timeline(transcribe_video(video_path, workspace_dir))
    vision = build_vision_timeline(analyze_video(video_path, workspace_dir))

    entries: list[dict] = []
    for candidate in approved_candidates:
        manifest = build_render_plan(
            candidate=candidate,
            source_video=video_path,
            transcript=transcript,
            vision=vision,
            probe=ingest.probe,
            workspace_dir=workspace_dir,
            ratios=tuple(job.output_profile.ratios),
            caption_preset=job.output_profile.caption_preset,
        )
        render_clip(manifest)
        manifest_path = render_manifest_path(workspace_dir, candidate.clip_id)
        write_json(manifest_path, manifest.model_dump(mode="json"))
        entries.append(
            {
                "clip_id": candidate.clip_id,
                "manifest_path": str(
                    manifest_path.relative_to(workspace_dir)
                    if manifest_path.is_relative_to(workspace_dir)
                    else manifest_path
                ),
                "outputs": {
                    ratio: str(
                        output.relative_to(workspace_dir)
                        if output.is_relative_to(workspace_dir)
                        else output
                    )
                    for ratio, output in manifest.outputs.items()
                },
                "render_dir": str(
                    clip_render_dir(workspace_dir, candidate.clip_id).relative_to(
                        workspace_dir
                    )
                ),
            }
        )

    report_path = workspace_dir / RENDER_REPORT_FILENAME
    write_json(
        report_path,
        {
            "job_id": ingest.job_id,
            "video_path": str(video_path),
            "clips": entries,
        },
    )
    return report_path


def _load_review_manifest(workspace_dir: Path) -> ReviewManifest | None:
    path = workspace_dir / REVIEW_MANIFEST_FILENAME
    if not path.exists():
        return None
    data = read_json(path)
    return ReviewManifest.model_validate(data)


def _brief_to_candidate(brief: ClipBrief) -> CandidateClip:
    return CandidateClip(
        clip_id=brief.clip_id,
        start_seconds=brief.start_seconds,
        end_seconds=brief.end_seconds,
        score=brief.mining_score,
        reasons=list(brief.mining_signals.reasons),
        spike_categories=list(brief.mining_signals.spike_categories),
    )


__all__ = [
    "MediaProbe",
    "RENDER_REPORT_FILENAME",
    "REVIEW_MANIFEST_FILENAME",
    "RenderStageError",
    "Stage",
    "VALID_STAGES",
    "analyze_video",
    "probe_video",
    "render_clip",
    "run_job",
    "score_shortlist",
    "transcribe_video",
]
