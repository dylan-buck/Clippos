import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from clippos.adapters.rubric import RUBRIC_VERSION
from clippos.models.scoring import ScoringResponse
from clippos.pipeline.scoring import SCORING_RESPONSE_FILENAME
from clippos.wrappers import claude_code, codex, common, hermes


def test_build_common_job_relies_on_model_defaults() -> None:
    captured: dict[str, str] = {}
    sentinel = object()

    def fake_model_validate(payload: dict[str, str]) -> object:
        captured.update(payload)
        return sentinel

    original = common.ClipposJob.model_validate
    common.ClipposJob.model_validate = fake_model_validate
    try:
        job = common.build_common_job("/tmp/input.mp4", "/tmp/out")
    finally:
        common.ClipposJob.model_validate = original

    assert job is sentinel
    assert captured == {
        "video_path": "/tmp/input.mp4",
        "output_dir": "/tmp/out",
    }


def test_codex_wrapper_delegates_to_common_helper(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    sentinel = object()

    def fake_build_common_job(video_path: str, output_dir: str) -> object:
        calls.append((video_path, output_dir))
        return sentinel

    monkeypatch.setattr(codex, "build_common_job", fake_build_common_job)

    job = codex.codex_job_from_args("/tmp/input.mp4", "/tmp/out")

    assert job is sentinel
    assert calls == [("/tmp/input.mp4", "/tmp/out")]


def test_claude_code_wrapper_delegates_to_common_helper(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    sentinel = object()

    def fake_build_common_job(video_path: str, output_dir: str) -> object:
        calls.append((video_path, output_dir))
        return sentinel

    monkeypatch.setattr(claude_code, "build_common_job", fake_build_common_job)

    job = claude_code.claude_job_from_args("/tmp/input.mp4", "/tmp/out")

    assert job is sentinel
    assert calls == [("/tmp/input.mp4", "/tmp/out")]


def test_hermes_wrapper_delegates_to_common_helper(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    sentinel = object()

    def fake_build_common_job(video_path: str, output_dir: str) -> object:
        calls.append((video_path, output_dir))
        return sentinel

    monkeypatch.setattr(hermes, "build_common_job", fake_build_common_job)

    job = hermes.hermes_job_from_args("/tmp/input.mp4", "/tmp/out")

    assert job is sentinel
    assert calls == [("/tmp/input.mp4", "/tmp/out")]


def test_load_workspace_scoring_request_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        common.load_workspace_scoring_request(tmp_path)


def _valid_response_payload() -> dict:
    return {
        "rubric_version": RUBRIC_VERSION,
        "job_id": "job-1",
        "scores": [
            {
                "clip_id": "clip-000",
                "clip_hash": "deadbeefdeadbeef",
                "rubric": {
                    "hook": 0.8,
                    "shareability": 0.7,
                    "standalone_clarity": 0.85,
                    "payoff": 0.75,
                    "delivery_energy": 0.65,
                    "quotability": 0.55,
                },
                "spike_categories": ["taboo"],
                "penalties": [],
                "final_score": 0.82,
                "title": "Secret reveal",
                "hook": "Nobody tells you this",
                "reasons": ["strong hook"],
            }
        ],
    }


def test_write_workspace_scoring_response_accepts_dict_payload(tmp_path: Path) -> None:
    payload = _valid_response_payload()
    path = common.write_workspace_scoring_response(tmp_path, payload)

    assert path == tmp_path / SCORING_RESPONSE_FILENAME
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["job_id"] == "job-1"
    assert data["scores"][0]["clip_id"] == "clip-000"


def test_write_workspace_scoring_response_accepts_model_instance(
    tmp_path: Path,
) -> None:
    response = ScoringResponse.model_validate(_valid_response_payload())
    path = common.write_workspace_scoring_response(tmp_path, response)
    assert path.exists()


def test_write_workspace_scoring_response_rejects_invalid_payload(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError):
        common.write_workspace_scoring_response(tmp_path, {"nonsense": True})


@pytest.mark.parametrize(
    "wrapper_module, loader_name, writer_name",
    [
        (claude_code, "claude_load_scoring_request", "claude_write_scoring_response"),
        (codex, "codex_load_scoring_request", "codex_write_scoring_response"),
        (hermes, "hermes_load_scoring_request", "hermes_write_scoring_response"),
    ],
)
def test_each_wrapper_delegates_scoring_helpers_to_common(
    monkeypatch, wrapper_module, loader_name: str, writer_name: str
) -> None:
    load_calls: list[Path] = []
    write_calls: list[tuple[Path, object]] = []
    load_sentinel = object()
    write_sentinel = object()

    def fake_load(workspace_dir: Path) -> object:
        load_calls.append(workspace_dir)
        return load_sentinel

    def fake_write(workspace_dir: Path, response: object) -> object:
        write_calls.append((workspace_dir, response))
        return write_sentinel

    monkeypatch.setattr(wrapper_module, "load_workspace_scoring_request", fake_load)
    monkeypatch.setattr(wrapper_module, "write_workspace_scoring_response", fake_write)

    loader = getattr(wrapper_module, loader_name)
    writer = getattr(wrapper_module, writer_name)

    assert loader(Path("/tmp/ws")) is load_sentinel
    assert writer(Path("/tmp/ws"), {"payload": True}) is write_sentinel
    assert load_calls == [Path("/tmp/ws")]
    assert write_calls == [(Path("/tmp/ws"), {"payload": True})]
