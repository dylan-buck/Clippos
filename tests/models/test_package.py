import pytest
from pydantic import ValidationError

from clippos.models.package import (
    MAX_HASHTAGS,
    MAX_HOOK_CHARS,
    MAX_HOOKS,
    MAX_THUMBNAIL_CHARS,
    MAX_TITLE_CHARS,
    MIN_HASHTAGS,
    MIN_HOOKS,
    MIN_THUMBNAIL_TEXTS,
    MIN_TITLES,
    PackageBrief,
    PublishPack,
)


def _valid_pack(**overrides) -> dict:
    payload: dict = {
        "clip_id": "clip-a",
        "clip_hash": "abc123",
        "titles": [f"Title variant {i}" for i in range(MIN_TITLES)],
        "thumbnail_texts": ["HOT TAKE", "BIG CLAIM", "DON'T MISS"],
        "social_caption": "A short caption body that fits within the limit.",
        "hashtags": [
            "#shorts",
            "#creator",
            "#podcast",
            "#viral",
            "#interview",
        ],
        "hooks": [
            "You will not believe what happens next.",
            "This one moment changed the interview.",
        ],
    }
    payload.update(overrides)
    return payload


def test_publish_pack_round_trip() -> None:
    pack = PublishPack.model_validate(_valid_pack())
    assert len(pack.titles) == MIN_TITLES
    assert len(pack.hashtags) == MIN_HASHTAGS
    assert len(pack.hooks) == MIN_HOOKS


def test_publish_pack_rejects_too_few_titles() -> None:
    with pytest.raises(ValidationError, match="at least"):
        PublishPack.model_validate(
            _valid_pack(titles=[f"Title {i}" for i in range(MIN_TITLES - 1)])
        )


def test_publish_pack_rejects_oversize_title() -> None:
    long_title = "x" * (MAX_TITLE_CHARS + 1)
    payload = _valid_pack()
    payload["titles"] = [long_title, *payload["titles"][1:]]
    with pytest.raises(ValidationError, match=f"{MAX_TITLE_CHARS} characters"):
        PublishPack.model_validate(payload)


def test_publish_pack_rejects_oversize_thumbnail_text() -> None:
    long_thumb = "X" * (MAX_THUMBNAIL_CHARS + 1)
    payload = _valid_pack()
    payload["thumbnail_texts"] = [
        long_thumb,
        *payload["thumbnail_texts"][1:],
    ]
    with pytest.raises(ValidationError, match=f"{MAX_THUMBNAIL_CHARS} characters"):
        PublishPack.model_validate(payload)


def test_publish_pack_enforces_hashtag_count_bounds() -> None:
    with pytest.raises(ValidationError, match=f"{MIN_HASHTAGS}"):
        PublishPack.model_validate(_valid_pack(hashtags=["#a", "#b", "#c", "#d"]))
    with pytest.raises(ValidationError, match=f"{MAX_HASHTAGS}"):
        PublishPack.model_validate(
            _valid_pack(hashtags=[f"#tag{i}" for i in range(MAX_HASHTAGS + 1)])
        )


def test_publish_pack_rejects_hashtag_without_hash_prefix() -> None:
    payload = _valid_pack()
    payload["hashtags"] = ["shorts", "#creator", "#podcast", "#viral", "#interview"]
    with pytest.raises(ValidationError, match="must start with '#'"):
        PublishPack.model_validate(payload)


def test_publish_pack_rejects_duplicate_hashtag_case_insensitive() -> None:
    payload = _valid_pack()
    payload["hashtags"] = ["#Shorts", "#shorts", "#podcast", "#viral", "#interview"]
    with pytest.raises(ValidationError, match="duplicated"):
        PublishPack.model_validate(payload)


def test_publish_pack_rejects_hashtag_with_whitespace() -> None:
    payload = _valid_pack()
    payload["hashtags"] = [
        "#big claim",
        "#creator",
        "#podcast",
        "#viral",
        "#interview",
    ]
    with pytest.raises(ValidationError, match="whitespace"):
        PublishPack.model_validate(payload)


def test_publish_pack_enforces_hook_count_bounds() -> None:
    with pytest.raises(ValidationError, match=f"{MIN_HOOKS}"):
        PublishPack.model_validate(_valid_pack(hooks=["only one"]))
    with pytest.raises(ValidationError, match=f"{MAX_HOOKS}"):
        PublishPack.model_validate(
            _valid_pack(hooks=[f"hook {i}" for i in range(MAX_HOOKS + 1)])
        )


def test_publish_pack_rejects_oversize_hook() -> None:
    long_hook = "h" * (MAX_HOOK_CHARS + 1)
    payload = _valid_pack()
    payload["hooks"] = [long_hook, "second hook that fits"]
    with pytest.raises(ValidationError, match=f"{MAX_HOOK_CHARS}"):
        PublishPack.model_validate(payload)


def test_publish_pack_rejects_empty_thumbnail_text() -> None:
    payload = _valid_pack()
    payload["thumbnail_texts"] = ["HOT", "  ", "OK"]
    with pytest.raises(ValidationError, match="empty"):
        PublishPack.model_validate(payload)


def test_publish_pack_rejects_empty_caption() -> None:
    with pytest.raises(ValidationError, match="empty"):
        PublishPack.model_validate(_valid_pack(social_caption="   "))


def test_publish_pack_respects_min_thumbnail_count() -> None:
    with pytest.raises(ValidationError, match=f"{MIN_THUMBNAIL_TEXTS}"):
        PublishPack.model_validate(_valid_pack(thumbnail_texts=["HOT", "BIG"]))


def test_package_brief_requires_positive_duration() -> None:
    with pytest.raises(ValidationError):
        PackageBrief.model_validate(
            {
                "clip_id": "clip-a",
                "clip_hash": "abc",
                "start_seconds": 10.0,
                "end_seconds": 10.0,
                "duration_seconds": 0.0,
                "transcript": "hello",
                "title_hint": "Title",
                "hook_hint": "Hook",
                "reasons": [],
                "spike_categories": [],
                "final_score": 0.5,
            }
        )
