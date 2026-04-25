import json
from pathlib import Path

import pytest

from clipper.models.job import ClipperJob
from clipper.models.render import RenderManifest
from clipper.pipeline.orchestrator import (
    RENDER_REPORT_FILENAME,
    REVIEW_MANIFEST_FILENAME,
    RenderStageError,
    run_job,
)
from clipper.pipeline.scoring import (
    SCORING_REQUEST_FILENAME,
    ScoringResponseError,
)

MOCK_TRANSCRIPT = {
    "segments": [
        {
            "speaker": "speaker_1",
            "start_seconds": 0.0,
            "end_seconds": 8.0,
            "text": "Here's the thing: this crazy secret will ban your favorite snack.",
            "words": [],
        },
        {
            "speaker": "speaker_1",
            "start_seconds": 8.0,
            "end_seconds": 18.0,
            "text": "Turns out the outcome was wild and the room erupted instantly.",
            "words": [],
        },
    ]
}

MOCK_VISION = {
    "frames": [
        {
            "timestamp_seconds": 1.0,
            "motion_score": 0.82,
            "shot_change": True,
            "primary_face": None,
        },
        {
            "timestamp_seconds": 10.0,
            "motion_score": 0.74,
            "shot_change": False,
            "primary_face": None,
        },
    ]
}


def _patch_pipeline_stages(
    monkeypatch: pytest.MonkeyPatch, observed_paths: list[Path]
) -> None:
    monkeypatch.setattr(
        "clipper.pipeline.orchestrator.probe_video",
        lambda path: (
            observed_paths.append(path)
            or {
                "duration_seconds": 120.0,
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "audio_sample_rate": 48000,
            }
        ),
    )
    monkeypatch.setattr(
        "clipper.pipeline.orchestrator.transcribe_video",
        lambda path, workspace: observed_paths.append(path) or MOCK_TRANSCRIPT,
    )
    monkeypatch.setattr(
        "clipper.pipeline.orchestrator.analyze_video",
        lambda path, workspace: observed_paths.append(path) or MOCK_VISION,
    )


def test_run_job_auto_stage_returns_review_manifest_when_scorer_resolves(
    sample_job: ClipperJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(sample_job.video_path.parent.resolve())
    expected_video_path = (Path.cwd() / sample_job.video_path).resolve(strict=False)
    observed_paths: list[Path] = []
    _patch_pipeline_stages(monkeypatch, observed_paths)

    mock_model_scores = [
        {
            "clip_id": "clip-000",
            "title": "Secret payoff",
            "hook": "Watch this reveal",
            "reasons": ["shareability", "payoff"],
            "final_score": 0.88,
        }
    ]
    monkeypatch.setattr(
        "clipper.pipeline.orchestrator.score_shortlist",
        lambda _workspace: mock_model_scores,
    )

    manifest_path = run_job(sample_job)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path.name == "review-manifest.json"
    assert manifest_path.exists()
    assert observed_paths == [
        expected_video_path,
        expected_video_path,
        expected_video_path,
    ]
    assert manifest["video_path"] == str(expected_video_path)
    assert manifest["candidates"][0]["title"] == "Secret payoff"
    assert manifest["candidates"][0]["hook"] == "Watch this reveal"
    assert manifest["candidates"][0]["reasons"] == ["shareability", "payoff"]
    assert manifest["candidates"][0]["score"] == 0.88


def test_run_job_auto_stage_returns_scoring_request_when_no_scores_available(
    sample_job: ClipperJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(sample_job.video_path.parent.resolve())
    observed_paths: list[Path] = []
    _patch_pipeline_stages(monkeypatch, observed_paths)

    artifact_path = run_job(sample_job)

    assert artifact_path.name == SCORING_REQUEST_FILENAME
    assert artifact_path.exists()


def test_run_job_mine_stage_writes_scoring_request_and_skips_review(
    sample_job: ClipperJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(sample_job.video_path.parent.resolve())
    observed_paths: list[Path] = []
    _patch_pipeline_stages(monkeypatch, observed_paths)

    score_shortlist_calls: list[Path] = []

    def _explode(workspace: Path) -> list[dict]:
        score_shortlist_calls.append(workspace)
        raise AssertionError("score_shortlist should not run during mine-only stage")

    monkeypatch.setattr("clipper.pipeline.orchestrator.score_shortlist", _explode)

    artifact_path = run_job(sample_job, stage="mine")

    assert artifact_path.name == SCORING_REQUEST_FILENAME
    assert artifact_path.exists()
    assert score_shortlist_calls == []


def test_run_job_review_stage_raises_when_no_scoring_context_exists(
    sample_job: ClipperJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(sample_job.video_path.parent.resolve())
    monkeypatch.setattr(
        "clipper.pipeline.orchestrator.probe_video",
        lambda _path: {
            "duration_seconds": 120.0,
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "audio_sample_rate": 48000,
        },
    )

    with pytest.raises(ScoringResponseError):
        run_job(sample_job, stage="review")


def test_run_job_review_stage_builds_manifest_from_existing_scoring_artifacts(
    sample_job: ClipperJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(sample_job.video_path.parent.resolve())
    observed_paths: list[Path] = []
    _patch_pipeline_stages(monkeypatch, observed_paths)

    # First run in mine stage to lay down the scoring-request artifact.
    run_job(sample_job, stage="mine")

    mock_model_scores = [
        {
            "clip_id": "clip-000",
            "title": "Review-stage title",
            "hook": "Review-stage hook",
            "reasons": ["payoff"],
            "final_score": 0.72,
        }
    ]
    monkeypatch.setattr(
        "clipper.pipeline.orchestrator.score_shortlist",
        lambda _workspace: mock_model_scores,
    )

    manifest_path = run_job(sample_job, stage="review")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path.name == "review-manifest.json"
    assert manifest["candidates"][0]["title"] == "Review-stage title"
    assert manifest["candidates"][0]["score"] == 0.72


def test_run_job_rejects_unknown_stage(sample_job: ClipperJob) -> None:
    with pytest.raises(ValueError):
        run_job(sample_job, stage="publish")  # type: ignore[arg-type]


def _prime_review_manifest(
    sample_job: ClipperJob, monkeypatch: pytest.MonkeyPatch
) -> Path:
    observed_paths: list[Path] = []
    _patch_pipeline_stages(monkeypatch, observed_paths)
    run_job(sample_job, stage="mine")
    mock_model_scores = [
        {
            "clip_id": "clip-000",
            "title": "Render-stage title",
            "hook": "Render-stage hook",
            "reasons": ["payoff"],
            "final_score": 0.81,
        }
    ]
    monkeypatch.setattr(
        "clipper.pipeline.orchestrator.score_shortlist",
        lambda _workspace: mock_model_scores,
    )
    return run_job(sample_job, stage="review")


def _approve_candidates(
    review_manifest_path: Path, approved_clip_ids: set[str]
) -> None:
    payload = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    for candidate in payload["candidates"]:
        candidate["approved"] = candidate["clip_id"] in approved_clip_ids
    review_manifest_path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_job_render_stage_without_review_manifest_raises(
    sample_job: ClipperJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(sample_job.video_path.parent.resolve())
    _patch_pipeline_stages(monkeypatch, [])

    with pytest.raises(RenderStageError):
        run_job(sample_job, stage="render")


def test_run_job_render_stage_without_approved_candidates_raises(
    sample_job: ClipperJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(sample_job.video_path.parent.resolve())
    review_manifest_path = _prime_review_manifest(sample_job, monkeypatch)
    assert review_manifest_path.name == REVIEW_MANIFEST_FILENAME

    def _explode(_manifest: RenderManifest) -> list:
        raise AssertionError("render should not be invoked without approvals")

    monkeypatch.setattr("clipper.pipeline.orchestrator.render_clip", _explode)

    with pytest.raises(RenderStageError, match="no approved candidates"):
        run_job(sample_job, stage="render")


def test_run_job_render_stage_writes_report_and_invokes_renderer(
    sample_job: ClipperJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(sample_job.video_path.parent.resolve())
    review_manifest_path = _prime_review_manifest(sample_job, monkeypatch)
    assert review_manifest_path.name == REVIEW_MANIFEST_FILENAME
    review_manifest = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    approved_clip_id = review_manifest["candidates"][0]["clip_id"]
    _approve_candidates(review_manifest_path, {approved_clip_id})

    rendered_manifests: list[RenderManifest] = []

    def fake_render_clip(manifest: RenderManifest) -> list:
        rendered_manifests.append(manifest)
        return []

    monkeypatch.setattr("clipper.pipeline.orchestrator.render_clip", fake_render_clip)

    report_path = run_job(sample_job, stage="render")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_path.name == RENDER_REPORT_FILENAME
    assert report["clips"]
    first_clip = report["clips"][0]
    assert "clip_id" in first_clip
    assert first_clip["manifest_path"].endswith("render-manifest.json")
    assert set(first_clip["outputs"].keys()) == {"9:16", "1:1", "16:9"}
    assert len(rendered_manifests) == len(report["clips"])
    assert all(manifest.approved for manifest in rendered_manifests)


def test_run_job_render_stage_honors_job_output_ratios(
    sample_job: ClipperJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(sample_job.video_path.parent.resolve())
    one_ratio_job = sample_job.model_copy(
        update={
            "output_profile": sample_job.output_profile.model_copy(
                update={"ratios": ["9:16"]}
            )
        }
    )
    review_manifest_path = _prime_review_manifest(one_ratio_job, monkeypatch)
    review_manifest = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    approved_clip_id = review_manifest["candidates"][0]["clip_id"]
    _approve_candidates(review_manifest_path, {approved_clip_id})

    rendered_manifests: list[RenderManifest] = []
    monkeypatch.setattr(
        "clipper.pipeline.orchestrator.render_clip",
        lambda manifest: rendered_manifests.append(manifest) or [],
    )

    report_path = run_job(one_ratio_job, stage="render")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert set(report["clips"][0]["outputs"].keys()) == {"9:16"}
    assert len(rendered_manifests) == 1
    assert set(rendered_manifests[0].outputs.keys()) == {"9:16"}


def test_run_job_auto_stage_does_not_chain_into_render(
    sample_job: ClipperJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(sample_job.video_path.parent.resolve())
    _patch_pipeline_stages(monkeypatch, [])
    monkeypatch.setattr(
        "clipper.pipeline.orchestrator.score_shortlist",
        lambda _workspace: [
            {
                "clip_id": "clip-000",
                "title": "Auto title",
                "hook": "Auto hook",
                "reasons": ["payoff"],
                "final_score": 0.7,
            }
        ],
    )

    def _explode(_manifest: RenderManifest) -> list:
        raise AssertionError("auto stage must not invoke the renderer")

    monkeypatch.setattr("clipper.pipeline.orchestrator.render_clip", _explode)

    artifact_path = run_job(sample_job)
    assert artifact_path.name == REVIEW_MANIFEST_FILENAME


# ---------- v1.1 brief stage integration ----------


def test_mine_stage_writes_brief_request_when_video_brief_enabled(
    sample_job: ClipperJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mine + video_brief=True writes BOTH scoring-request.json and
    brief-request.json so the harness state machine can pick up the
    brief handoff next."""
    monkeypatch.chdir(sample_job.video_path.parent.resolve())
    observed_paths: list[Path] = []
    _patch_pipeline_stages(monkeypatch, observed_paths)

    job_with_brief = sample_job.model_copy(
        update={
            "output_profile": sample_job.output_profile.model_copy(
                update={"video_brief": True}
            )
        }
    )
    request_path = run_job(job_with_brief, stage="mine")
    workspace = request_path.parent

    assert (workspace / "scoring-request.json").exists()
    assert (workspace / "brief-request.json").exists(), (
        "video_brief=True must produce brief-request.json from the mine stage"
    )


def test_mine_stage_skips_brief_request_when_video_brief_disabled(
    sample_job: ClipperJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    """video_brief=False (the test fixture default) preserves the
    legacy mine flow — no brief-request.json is written."""
    monkeypatch.chdir(sample_job.video_path.parent.resolve())
    _patch_pipeline_stages(monkeypatch, [])

    request_path = run_job(sample_job, stage="mine")
    workspace = request_path.parent

    assert (workspace / "scoring-request.json").exists()
    assert not (workspace / "brief-request.json").exists()


def test_auto_stage_pauses_for_brief_when_brief_enabled_and_unauthored(
    sample_job: ClipperJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto must NOT proceed to scoring when video_brief is enabled
    but the harness has not yet authored brief-response.json. Returns
    the brief request path so the caller knows what to wait on."""
    monkeypatch.chdir(sample_job.video_path.parent.resolve())
    _patch_pipeline_stages(monkeypatch, [])

    score_calls: list[Path] = []

    def _explode(workspace: Path) -> list[dict]:
        score_calls.append(workspace)
        raise AssertionError(
            "score_shortlist must not run while waiting on the brief handoff"
        )

    monkeypatch.setattr("clipper.pipeline.orchestrator.score_shortlist", _explode)

    job_with_brief = sample_job.model_copy(
        update={
            "output_profile": sample_job.output_profile.model_copy(
                update={"video_brief": True}
            )
        }
    )
    artifact_path = run_job(job_with_brief)

    assert artifact_path.name == "brief-request.json"
    assert score_calls == []


def test_auto_stage_embeds_brief_into_scoring_request_when_response_exists(
    sample_job: ClipperJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the brief response is already on disk, auto picks it up,
    embeds it into scoring-request.json, and proceeds to scoring."""
    from clipper.adapters.brief import BRIEF_VERSION
    from clipper.models.scoring import VideoBrief, VideoBriefResponse
    from clipper.pipeline.brief import brief_response_path
    from clipper.pipeline.scoring import load_scoring_request

    monkeypatch.chdir(sample_job.video_path.parent.resolve())
    _patch_pipeline_stages(monkeypatch, [])

    job_with_brief = sample_job.model_copy(
        update={
            "output_profile": sample_job.output_profile.model_copy(
                update={"video_brief": True}
            )
        }
    )
    # Step 1: run mine to lay down scoring-request.json + brief-request.json.
    request_path = run_job(job_with_brief, stage="mine")
    workspace = request_path.parent

    # Step 2: simulate the harness writing brief-response.json.
    scoring_request = load_scoring_request(workspace)
    assert scoring_request is not None
    assert scoring_request.video_brief is None  # no brief embedded yet
    original_clip_hashes = [clip.clip_hash for clip in scoring_request.clips]

    fake_brief = VideoBrief(
        rubric_version=BRIEF_VERSION,
        job_id=scoring_request.job_id,
        theme="A staged brief that the test embeds.",
        video_format="podcast interview",
        expected_viral_patterns=[
            "mocked pattern",
            "specific guest payoff",
            "clear thesis moment",
        ],
    )
    response = VideoBriefResponse(
        rubric_version=BRIEF_VERSION,
        job_id=scoring_request.job_id,
        brief=fake_brief,
    )
    brief_response_path(workspace).write_text(
        json.dumps(response.model_dump(mode="json")),
        encoding="utf-8",
    )

    # Step 3: stub the scoring step + run auto. Should embed the brief
    # and proceed to review.
    monkeypatch.setattr(
        "clipper.pipeline.orchestrator.score_shortlist",
        lambda _workspace: [
            {
                "clip_id": "clip-000",
                "title": "After-brief title",
                "hook": "After-brief hook",
                "reasons": ["payoff"],
                "final_score": 0.74,
            }
        ],
    )
    artifact_path = run_job(job_with_brief)

    assert artifact_path.name == REVIEW_MANIFEST_FILENAME
    # And scoring-request.json was rewritten to embed the brief.
    rewritten = load_scoring_request(workspace)
    assert rewritten is not None
    assert rewritten.video_brief == fake_brief
    assert [clip.clip_hash for clip in rewritten.clips] != original_clip_hashes


def test_brief_stage_raises_when_no_response_or_cache_available(
    sample_job: ClipperJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Standalone --stage brief must fail clearly when there is nothing
    to resolve (no response, no cache) — silently re-running mine would
    waste minutes of compute."""
    from clipper.pipeline.brief import BriefResponseError

    monkeypatch.chdir(sample_job.video_path.parent.resolve())
    _patch_pipeline_stages(monkeypatch, [])

    job_with_brief = sample_job.model_copy(
        update={
            "output_profile": sample_job.output_profile.model_copy(
                update={"video_brief": True}
            )
        }
    )
    run_job(job_with_brief, stage="mine")  # writes brief-request.json

    with pytest.raises(BriefResponseError):
        run_job(job_with_brief, stage="brief")
