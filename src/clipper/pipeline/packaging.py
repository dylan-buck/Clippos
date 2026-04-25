"""Packaging stage: turn approved clips into a publish pack.

The pipeline here mirrors the scoring handoff: the clipper assembles a
``package-request.json`` with per-clip context plus the prompt + response
schema, the surrounding harness writes ``package-response.json`` with one
``PublishPack`` per clip, and ``package-save`` validates the response and
fans the packs out next to their rendered MP4s. The clipper itself never
calls an LLM directly.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from clipper.adapters.storage import read_json, write_json
from clipper.models.candidate import CandidateClip
from clipper.models.package import (
    MAX_CAPTION_CHARS,
    MAX_HASHTAGS,
    MAX_HOOK_CHARS,
    MAX_HOOKS,
    MAX_THUMBNAIL_CHARS,
    MAX_TITLE_CHARS,
    MIN_HASHTAGS,
    MIN_HOOKS,
    MIN_THUMBNAIL_TEXTS,
    MIN_TITLES,
    PACKAGE_PROMPT_VERSION,
    PackageBrief,
    PackageRequest,
    PackageResponse,
    PublishPack,
)
from clipper.models.review import ReviewManifest
from clipper.models.scoring import ClipBrief, ScoringRequest, VideoBrief

PACKAGE_REQUEST_FILENAME = "package-request.json"
PACKAGE_RESPONSE_FILENAME = "package-response.json"
PACKAGE_REPORT_FILENAME = "package-report.json"
CLIP_PACKAGE_FILENAME = "package.json"


class PackagingResponseError(RuntimeError):
    pass


def package_request_path(workspace_dir: Path) -> Path:
    return workspace_dir / PACKAGE_REQUEST_FILENAME


def package_response_path(workspace_dir: Path) -> Path:
    return workspace_dir / PACKAGE_RESPONSE_FILENAME


def package_report_path(workspace_dir: Path) -> Path:
    return workspace_dir / PACKAGE_REPORT_FILENAME


def build_package_prompt() -> str:
    return (
        "You are packaging a rendered short-form video clip for publishing on "
        "TikTok, YouTube Shorts, and Instagram Reels. For every clip in the "
        "request you must return a PublishPack that is ready to paste into the "
        "upload forms.\n\n"
        "Use the clip transcript, title_hint, hook_hint, reasons, and "
        "spike_categories as your only signal. Do not invent facts the transcript "
        "cannot support. Never reuse the exact title_hint — each title must be a "
        "distinct rewrite or angle.\n\n"
        f"For each clip return exactly:\n"
        f"- titles: at least {MIN_TITLES} candidate titles, each ≤ "
        f"{MAX_TITLE_CHARS} characters, distinct angles/framings.\n"
        f"- thumbnail_texts: at least {MIN_THUMBNAIL_TEXTS} short overlay lines, "
        f"each ≤ {MAX_THUMBNAIL_CHARS} characters, high-contrast and scannable.\n"
        f"- social_caption: one caption body, ≤ {MAX_CAPTION_CHARS} characters, "
        f"2–4 sentences, no hashtags inside.\n"
        f"- hashtags: between {MIN_HASHTAGS} and {MAX_HASHTAGS} hashtags, each "
        f"prefixed with '#', no spaces, no duplicates.\n"
        f"- hooks: between {MIN_HOOKS} and {MAX_HOOKS} opening-line rewrites, "
        f"each ≤ {MAX_HOOK_CHARS} characters, designed to survive the 2-second "
        f"scroll test.\n\n"
        "Return the packs in the same order as the request, preserving clip_id "
        "and clip_hash verbatim. Response must match the attached response_schema."
    )


def build_package_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["prompt_version", "job_id", "packs"],
        "properties": {
            "prompt_version": {"type": "string"},
            "job_id": {"type": "string"},
            "packs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "clip_id",
                        "clip_hash",
                        "titles",
                        "thumbnail_texts",
                        "social_caption",
                        "hashtags",
                        "hooks",
                    ],
                    "properties": {
                        "clip_id": {"type": "string"},
                        "clip_hash": {"type": "string"},
                        "titles": {
                            "type": "array",
                            "minItems": MIN_TITLES,
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": MAX_TITLE_CHARS,
                            },
                        },
                        "thumbnail_texts": {
                            "type": "array",
                            "minItems": MIN_THUMBNAIL_TEXTS,
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": MAX_THUMBNAIL_CHARS,
                            },
                        },
                        "social_caption": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_CAPTION_CHARS,
                        },
                        "hashtags": {
                            "type": "array",
                            "minItems": MIN_HASHTAGS,
                            "maxItems": MAX_HASHTAGS,
                            "items": {
                                "type": "string",
                                "pattern": "^#[^\\s#]+$",
                            },
                        },
                        "hooks": {
                            "type": "array",
                            "minItems": MIN_HOOKS,
                            "maxItems": MAX_HOOKS,
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": MAX_HOOK_CHARS,
                            },
                        },
                    },
                },
            },
        },
    }


def build_package_brief(
    *, candidate: CandidateClip, brief: ClipBrief, final_score: float | None = None
) -> PackageBrief:
    return PackageBrief(
        clip_id=candidate.clip_id,
        clip_hash=brief.clip_hash,
        start_seconds=candidate.start_seconds,
        end_seconds=candidate.end_seconds,
        duration_seconds=candidate.end_seconds - candidate.start_seconds,
        transcript=brief.transcript,
        title_hint=candidate.title,
        hook_hint=candidate.hook,
        reasons=list(candidate.reasons),
        spike_categories=list(candidate.spike_categories),
        final_score=candidate.score if final_score is None else final_score,
    )


def build_package_request(
    *,
    job_id: str,
    video_path: Path,
    briefs: list[PackageBrief],
    prompt_version: str = PACKAGE_PROMPT_VERSION,
    video_brief: VideoBrief | None = None,
) -> PackageRequest:
    if not briefs:
        raise ValueError("package request requires at least one brief")
    return PackageRequest(
        prompt_version=prompt_version,
        job_id=job_id,
        video_path=video_path,
        package_prompt=build_package_prompt(),
        response_schema=build_package_schema(),
        clips=briefs,
        video_brief=video_brief,
    )


def briefs_for_approved_candidates(
    review: ReviewManifest, scoring_request: ScoringRequest
) -> list[PackageBrief]:
    briefs_by_clip_id = {brief.clip_id: brief for brief in scoring_request.clips}
    collected: list[PackageBrief] = []
    for candidate in review.candidates:
        if not candidate.approved:
            continue
        brief = briefs_by_clip_id.get(candidate.clip_id)
        if brief is None:
            raise PackagingResponseError(
                f"scoring-request.json missing clip_id {candidate.clip_id!r} "
                "referenced by an approved review candidate"
            )
        collected.append(build_package_brief(candidate=candidate, brief=brief))
    return collected


def write_package_request(workspace_dir: Path, request: PackageRequest) -> Path:
    path = package_request_path(workspace_dir)
    write_json(path, request.model_dump(mode="json"))
    return path


def load_package_request(workspace_dir: Path) -> PackageRequest | None:
    path = package_request_path(workspace_dir)
    if not path.exists():
        return None
    try:
        data = read_json(path)
    except (json.JSONDecodeError, OSError):
        return None
    try:
        return PackageRequest.model_validate(data)
    except ValidationError:
        return None


def load_package_response(workspace_dir: Path) -> PackageResponse:
    path = package_response_path(workspace_dir)
    if not path.exists():
        raise PackagingResponseError(
            f"{PACKAGE_RESPONSE_FILENAME} not found in workspace"
        )
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PackagingResponseError(
            f"Unable to read {PACKAGE_RESPONSE_FILENAME}"
        ) from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PackagingResponseError(
            f"{PACKAGE_RESPONSE_FILENAME} is not valid JSON"
        ) from exc
    try:
        return PackageResponse.model_validate(data)
    except ValidationError as exc:
        raise PackagingResponseError(
            f"{PACKAGE_RESPONSE_FILENAME} does not match packaging contract"
        ) from exc


def resolve_packs(
    request: PackageRequest, response: PackageResponse
) -> list[PublishPack]:
    if response.prompt_version != request.prompt_version:
        raise PackagingResponseError(
            f"{PACKAGE_RESPONSE_FILENAME} prompt_version "
            f"{response.prompt_version!r} != request {request.prompt_version!r}"
        )
    if response.job_id != request.job_id:
        raise PackagingResponseError(
            f"{PACKAGE_RESPONSE_FILENAME} job_id {response.job_id!r} != request "
            f"{request.job_id!r}"
        )
    packs_by_hash = {pack.clip_hash: pack for pack in response.packs}
    resolved: list[PublishPack] = []
    for brief in request.clips:
        pack = packs_by_hash.get(brief.clip_hash)
        if pack is None:
            raise PackagingResponseError(
                f"{PACKAGE_RESPONSE_FILENAME} missing pack for clip_hash "
                f"{brief.clip_hash!r}"
            )
        if pack.clip_id != brief.clip_id:
            raise PackagingResponseError(
                f"{PACKAGE_RESPONSE_FILENAME} clip_hash {brief.clip_hash!r} "
                f"returned clip_id {pack.clip_id!r}, expected {brief.clip_id!r}"
            )
        resolved.append(pack)
    return resolved


def write_pack_artifacts(
    *,
    workspace_dir: Path,
    clip_dir_for: Callable[[str], Path],
    packs: list[PublishPack],
) -> dict[str, Path]:
    written: dict[str, Path] = {}
    for pack in packs:
        target_dir = clip_dir_for(pack.clip_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / CLIP_PACKAGE_FILENAME
        write_json(target, pack.model_dump(mode="json"))
        written[pack.clip_id] = target
    return written


def write_package_report(
    *, workspace_dir: Path, job_id: str, video_path: Path, pack_paths: dict[str, Path]
) -> Path:
    payload = {
        "job_id": job_id,
        "video_path": str(video_path),
        "packs": [
            {
                "clip_id": clip_id,
                "pack_path": str(
                    path.relative_to(workspace_dir)
                    if path.is_relative_to(workspace_dir)
                    else path
                ),
            }
            for clip_id, path in pack_paths.items()
        ],
    }
    report = package_report_path(workspace_dir)
    write_json(report, payload)
    return report
