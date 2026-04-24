from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator, model_validator

from clipper.models.media import ContractModel

PACKAGE_PROMPT_VERSION = "package-v1"

MIN_TITLES = 5
MAX_TITLE_CHARS = 80

MIN_THUMBNAIL_TEXTS = 3
MAX_THUMBNAIL_CHARS = 28

MIN_HASHTAGS = 5
MAX_HASHTAGS = 10

MIN_HOOKS = 2
MAX_HOOKS = 3
MAX_HOOK_CHARS = 140

MAX_CAPTION_CHARS = 500


class PackageBrief(ContractModel):
    """Per-clip context handed to the harness for packaging."""

    clip_id: str
    clip_hash: str
    start_seconds: Annotated[float, Field(ge=0)]
    end_seconds: Annotated[float, Field(ge=0)]
    duration_seconds: Annotated[float, Field(gt=0)]
    transcript: str
    title_hint: str
    hook_hint: str
    reasons: list[str]
    spike_categories: list[str]
    final_score: Annotated[float, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def validate_time_bounds(self) -> "PackageBrief":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class PackageRequest(ContractModel):
    prompt_version: str
    job_id: str
    video_path: Path
    package_prompt: str
    response_schema: dict
    clips: list[PackageBrief]


class PublishPack(ContractModel):
    """Harness-authored publish pack for a single rendered clip."""

    clip_id: str
    clip_hash: str
    titles: list[str]
    thumbnail_texts: list[str]
    social_caption: str
    hashtags: list[str]
    hooks: list[str]

    @field_validator("titles")
    @classmethod
    def validate_titles(cls, value: list[str]) -> list[str]:
        if len(value) < MIN_TITLES:
            raise ValueError(f"titles must include at least {MIN_TITLES} entries")
        for title in value:
            stripped = title.strip()
            if not stripped:
                raise ValueError("titles must not contain empty entries")
            if len(stripped) > MAX_TITLE_CHARS:
                raise ValueError(
                    f"title {stripped!r} exceeds {MAX_TITLE_CHARS} characters"
                )
        return value

    @field_validator("thumbnail_texts")
    @classmethod
    def validate_thumbnail_texts(cls, value: list[str]) -> list[str]:
        if len(value) < MIN_THUMBNAIL_TEXTS:
            raise ValueError(
                f"thumbnail_texts must include at least {MIN_THUMBNAIL_TEXTS} entries"
            )
        for text in value:
            stripped = text.strip()
            if not stripped:
                raise ValueError("thumbnail_texts must not contain empty entries")
            if len(stripped) > MAX_THUMBNAIL_CHARS:
                raise ValueError(
                    f"thumbnail text {stripped!r} exceeds "
                    f"{MAX_THUMBNAIL_CHARS} characters"
                )
        return value

    @field_validator("social_caption")
    @classmethod
    def validate_social_caption(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("social_caption must not be empty")
        if len(stripped) > MAX_CAPTION_CHARS:
            raise ValueError(f"social_caption exceeds {MAX_CAPTION_CHARS} characters")
        return value

    @field_validator("hashtags")
    @classmethod
    def validate_hashtags(cls, value: list[str]) -> list[str]:
        if not MIN_HASHTAGS <= len(value) <= MAX_HASHTAGS:
            raise ValueError(
                f"hashtags must include between {MIN_HASHTAGS} and {MAX_HASHTAGS} entries"
            )
        seen: set[str] = set()
        for tag in value:
            stripped = tag.strip()
            if not stripped:
                raise ValueError("hashtags must not contain empty entries")
            if not stripped.startswith("#"):
                raise ValueError(f"hashtag {stripped!r} must start with '#'")
            if " " in stripped:
                raise ValueError(f"hashtag {stripped!r} must not contain whitespace")
            lowered = stripped.lower()
            if lowered in seen:
                raise ValueError(f"hashtag {stripped!r} is duplicated")
            seen.add(lowered)
        return value

    @field_validator("hooks")
    @classmethod
    def validate_hooks(cls, value: list[str]) -> list[str]:
        if not MIN_HOOKS <= len(value) <= MAX_HOOKS:
            raise ValueError(
                f"hooks must include between {MIN_HOOKS} and {MAX_HOOKS} entries"
            )
        for hook in value:
            stripped = hook.strip()
            if not stripped:
                raise ValueError("hooks must not contain empty entries")
            if len(stripped) > MAX_HOOK_CHARS:
                raise ValueError(
                    f"hook {stripped!r} exceeds {MAX_HOOK_CHARS} characters"
                )
        return value


class PackageResponse(ContractModel):
    prompt_version: str
    job_id: str
    packs: list[PublishPack]


__all__ = [
    "MAX_CAPTION_CHARS",
    "MAX_HASHTAGS",
    "MAX_HOOKS",
    "MAX_HOOK_CHARS",
    "MAX_THUMBNAIL_CHARS",
    "MAX_TITLE_CHARS",
    "MIN_HASHTAGS",
    "MIN_HOOKS",
    "MIN_THUMBNAIL_TEXTS",
    "MIN_TITLES",
    "PACKAGE_PROMPT_VERSION",
    "PackageBrief",
    "PackageRequest",
    "PackageResponse",
    "PublishPack",
]
