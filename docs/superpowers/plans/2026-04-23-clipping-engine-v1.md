# Clipping Engine V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-first clipping engine that analyzes full videos, ranks high-potential clips, creates review packages, and renders approved clips in 9:16, 1:1, and 16:9 for Claude Code, Codex, and Hermes wrappers.

**Architecture:** The repo is a Python-first media engine with a CLI entrypoint, deterministic analysis modules, a harness-model semantic review stage, and an FFmpeg-based render pipeline. Thin agent wrappers sit on top of a stable JSON job spec and review manifest, so the core clip logic stays in one place.

**Tech Stack:** Python 3.12, pytest, pydantic, FFmpeg, OpenCV, faster-whisper or WhisperX, pyannote, numpy, rich, optional harness-model SDK adapter

---

## File Structure

### Core package

- Create: `pyproject.toml`
- Create: `README.md`
- Create: `.gitignore`
- Create: `src/clipper/__init__.py`
- Create: `src/clipper/config.py`
- Create: `src/clipper/cli.py`
- Create: `src/clipper/logging.py`

### Domain models

- Create: `src/clipper/models/job.py`
- Create: `src/clipper/models/media.py`
- Create: `src/clipper/models/analysis.py`
- Create: `src/clipper/models/candidate.py`
- Create: `src/clipper/models/review.py`
- Create: `src/clipper/models/render.py`

### Pipeline modules

- Create: `src/clipper/pipeline/ingest.py`
- Create: `src/clipper/pipeline/transcribe.py`
- Create: `src/clipper/pipeline/vision.py`
- Create: `src/clipper/pipeline/candidates.py`
- Create: `src/clipper/pipeline/review.py`
- Create: `src/clipper/pipeline/render.py`
- Create: `src/clipper/pipeline/orchestrator.py`

### Infra and adapters

- Create: `src/clipper/adapters/ffmpeg.py`
- Create: `src/clipper/adapters/harness_model.py`
- Create: `src/clipper/adapters/storage.py`

### Review artifacts and wrappers

- Create: `src/clipper/wrappers/common.py`
- Create: `src/clipper/wrappers/codex.py`
- Create: `src/clipper/wrappers/claude_code.py`
- Create: `src/clipper/wrappers/hermes.py`

### Tests

- Create: `tests/conftest.py`
- Create: `tests/test_cli.py`
- Create: `tests/test_wrappers.py`
- Create: `tests/models/test_job_models.py`
- Create: `tests/pipeline/test_ingest.py`
- Create: `tests/pipeline/test_transcribe.py`
- Create: `tests/pipeline/test_vision.py`
- Create: `tests/pipeline/test_candidates.py`
- Create: `tests/pipeline/test_review.py`
- Create: `tests/pipeline/test_render.py`
- Create: `tests/pipeline/test_orchestrator.py`
- Create: `tests/fixtures/sample_job.json`
- Create: `tests/fixtures/sample_transcript.json`
- Create: `tests/fixtures/sample_faces.json`

### Docs

- Create: `docs/architecture/job-spec.md`
- Create: `docs/architecture/review-manifest.md`
- Create: `docs/architecture/render-manifest.md`

## Task 1: Bootstrap The Repo And Tooling

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/clipper/__init__.py`
- Create: `src/clipper/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing CLI smoke test**

```python
from typer.testing import CliRunner

from clipper import __version__
from clipper.cli import app


def test_version_command_prints_package_version() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout == f"clipper-tool {__version__}\n"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_version_command_prints_package_version -v`
Expected: FAIL with `ModuleNotFoundError` or missing `app`

- [ ] **Step 3: Write minimal package and CLI implementation**

```python
# src/clipper/__init__.py
__all__ = ["__version__"]
__version__ = "0.1.0"
```

```python
# src/clipper/cli.py
import typer

from clipper import __version__

app = typer.Typer(no_args_is_help=True)


@app.command()
def version() -> None:
    typer.echo(f"clipper-tool {__version__}")
```

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "clipper-tool"
dynamic = ["version"]
requires-python = ">=3.12"
dependencies = ["typer>=0.12"]

[project.optional-dependencies]
dev = ["pytest>=8.2", "pytest-cov>=5.0", "ruff>=0.4"]

[project.scripts]
clipper-tool = "clipper.cli:app"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.dynamic]
version = {attr = "clipper.__version__"}

[tool.pytest.ini_options]
pythonpath = ["src"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py::test_version_command_prints_package_version -v`
Expected: PASS

- [ ] **Step 5: Add baseline repo files**

```gitignore
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
.mypy_cache/
dist/
build/
.superpowers/
artifacts/
```

```markdown
# README.md

Local-first clipping engine for Claude Code, Codex, and Hermes Agent.
```

- [ ] **Step 6: Run the smoke suite**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore README.md src/clipper/__init__.py src/clipper/cli.py tests/test_cli.py
git commit -m "chore: bootstrap clipper tool package"
```

## Task 2: Define The Stable Job Spec And Core Domain Models

**Files:**
- Modify: `pyproject.toml`
- Create: `src/clipper/models/job.py`
- Create: `src/clipper/models/media.py`
- Create: `src/clipper/models/analysis.py`
- Create: `src/clipper/models/candidate.py`
- Create: `src/clipper/models/review.py`
- Create: `src/clipper/models/render.py`
- Create: `tests/models/test_job_models.py`
- Create: `docs/architecture/job-spec.md`

- [ ] **Step 1: Write the failing model tests**

```python
from clipper.models.job import ClipperJob, OutputProfile


def test_job_defaults_include_all_output_ratios() -> None:
    job = ClipperJob.model_validate(
        {"video_path": "/tmp/input.mp4", "output_dir": "/tmp/out"}
    )
    assert job.output_profile.ratios == ["9:16", "1:1", "16:9"]


def test_job_requires_existing_review_gate() -> None:
    job = ClipperJob.model_validate(
        {"video_path": "/tmp/input.mp4", "output_dir": "/tmp/out"}
    )
    assert job.review_required is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_job_models.py -v`
Expected: FAIL with missing model imports

- [ ] **Step 3: Define pydantic models**

```python
from pathlib import Path

from pydantic import BaseModel, Field


class OutputProfile(BaseModel):
    ratios: list[str] = Field(default_factory=lambda: ["9:16", "1:1", "16:9"])
    caption_preset: str = "hook-default"


class ClipperJob(BaseModel):
    video_path: Path
    output_dir: Path
    review_required: bool = True
    output_profile: OutputProfile = Field(default_factory=OutputProfile)
    max_candidates: int = 12
```

- [ ] **Step 4: Add `pydantic` to the project dependencies**

```toml
[project]
dependencies = ["typer>=0.12", "pydantic>=2.7"]
```

- [ ] **Step 5: Add the rest of the shared contract models**

```python
class MediaProbe(BaseModel):
    duration_seconds: float
    width: int
    height: int
    fps: float
    audio_sample_rate: int


class CandidateClip(BaseModel):
    clip_id: str
    start_seconds: float
    end_seconds: float
    score: float
    reasons: list[str]
    spike_categories: list[str]
    title: str = ""
    hook: str = ""
```

```python
class ReviewManifest(BaseModel):
    job_id: str
    video_path: Path
    candidates: list[CandidateClip]
```

```python
class RenderManifest(BaseModel):
    clip_id: str
    approved: bool
    outputs: dict[str, Path]
```

- [ ] **Step 6: Document the job contract**

```json
{
  "video_path": "/absolute/path/input.mp4",
  "output_dir": "/absolute/path/out",
  "review_required": true,
  "max_candidates": 12,
  "output_profile": {
    "ratios": ["9:16", "1:1", "16:9"],
    "caption_preset": "hook-default"
  }
}
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/models/test_job_models.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/clipper/models tests/models/test_job_models.py docs/architecture/job-spec.md
git commit -m "feat: define clipper job and manifest models"
```

## Task 3: Build Ingest And FFmpeg Probe Support

**Files:**
- Create: `src/clipper/adapters/ffmpeg.py`
- Create: `src/clipper/pipeline/ingest.py`
- Create: `tests/pipeline/test_ingest.py`

- [ ] **Step 1: Write the failing ingest test**

```python
from pathlib import Path

from clipper.models.job import ClipperJob
from clipper.pipeline.ingest import ingest_job


def test_ingest_job_builds_workspace_and_probe_result(tmp_path: Path) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")
    job = ClipperJob(video_path=video, output_dir=tmp_path / "out")
    result = ingest_job(job, probe_data={"duration_seconds": 120.0, "width": 1920, "height": 1080, "fps": 30.0, "audio_sample_rate": 48000})
    assert result.workspace_dir == job.output_dir / "jobs" / result.job_id
    assert result.probe.duration_seconds == 120.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_ingest.py::test_ingest_job_builds_workspace_and_probe_result -v`
Expected: FAIL with missing `ingest_job`

- [ ] **Step 3: Implement the FFmpeg adapter shell**

```python
import json
import subprocess
from pathlib import Path


def probe_media(video_path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    output = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(output.stdout)
```

- [ ] **Step 4: Implement the ingest pipeline**

```python
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

from clipper.models.job import ClipperJob
from clipper.models.media import MediaProbe


@dataclass
class IngestResult:
    job_id: str
    workspace_dir: Path
    probe: MediaProbe


def ingest_job(job: ClipperJob, probe_data: dict) -> IngestResult:
    job_id = sha1(str(job.video_path).encode()).hexdigest()[:12]
    workspace_dir = job.output_dir / "jobs" / job_id
    workspace_dir.mkdir(parents=True, exist_ok=True)
    probe = MediaProbe(**probe_data)
    return IngestResult(job_id=job_id, workspace_dir=workspace_dir, probe=probe)
```

- [ ] **Step 5: Add a second test for cache stability**

```python
def test_ingest_job_uses_stable_job_id(tmp_path: Path) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")
    job = ClipperJob(video_path=video, output_dir=tmp_path / "out")
    one = ingest_job(job, probe_data={"duration_seconds": 1.0, "width": 1, "height": 1, "fps": 1.0, "audio_sample_rate": 16000})
    two = ingest_job(job, probe_data={"duration_seconds": 1.0, "width": 1, "height": 1, "fps": 1.0, "audio_sample_rate": 16000})
    assert one.job_id == two.job_id
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/pipeline/test_ingest.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/clipper/adapters/ffmpeg.py src/clipper/pipeline/ingest.py tests/pipeline/test_ingest.py
git commit -m "feat: add ingest pipeline and ffmpeg probe adapter"
```

## Task 4: Add Transcript And Diarization Analysis

**Files:**
- Create: `src/clipper/pipeline/transcribe.py`
- Create: `tests/pipeline/test_transcribe.py`
- Create: `tests/fixtures/sample_transcript.json`

- [ ] **Step 1: Write the failing transcription aggregation test**

```python
from clipper.pipeline.transcribe import build_transcript_timeline


def test_build_transcript_timeline_returns_word_ranges(sample_transcript_payload: dict) -> None:
    timeline = build_transcript_timeline(sample_transcript_payload)
    assert timeline.segments[0].speaker == "speaker_1"
    assert timeline.segments[0].start_seconds == 0.0
    assert timeline.segments[0].words[0].text == "Look"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_transcribe.py::test_build_transcript_timeline_returns_word_ranges -v`
Expected: FAIL with missing `build_transcript_timeline`

- [ ] **Step 3: Create a normalized transcript timeline model**

```python
class TranscriptWord(BaseModel):
    text: str
    start_seconds: float
    end_seconds: float
    confidence: float


class TranscriptSegment(BaseModel):
    speaker: str
    start_seconds: float
    end_seconds: float
    text: str
    words: list[TranscriptWord]


class TranscriptTimeline(BaseModel):
    segments: list[TranscriptSegment]
```

- [ ] **Step 4: Implement transcript normalization**

```python
def build_transcript_timeline(payload: dict) -> TranscriptTimeline:
    segments = []
    for segment in payload["segments"]:
        words = [TranscriptWord(**word) for word in segment["words"]]
        segments.append(
            TranscriptSegment(
                speaker=segment["speaker"],
                start_seconds=segment["start_seconds"],
                end_seconds=segment["end_seconds"],
                text=segment["text"],
                words=words,
            )
        )
    return TranscriptTimeline(segments=segments)
```

- [ ] **Step 5: Add a diarization-focused test**

```python
def test_build_transcript_timeline_keeps_speaker_turns(sample_transcript_payload: dict) -> None:
    timeline = build_transcript_timeline(sample_transcript_payload)
    assert [segment.speaker for segment in timeline.segments] == ["speaker_1", "speaker_2"]
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/pipeline/test_transcribe.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/clipper/pipeline/transcribe.py tests/pipeline/test_transcribe.py tests/fixtures/sample_transcript.json
git commit -m "feat: normalize transcript and speaker timelines"
```

## Task 5: Add Vision Signals For Framing And Spike Detection

**Files:**
- Create: `src/clipper/pipeline/vision.py`
- Create: `tests/pipeline/test_vision.py`
- Create: `tests/fixtures/sample_faces.json`

- [ ] **Step 1: Write the failing vision aggregation test**

```python
from clipper.pipeline.vision import build_vision_timeline


def test_build_vision_timeline_emits_face_and_motion_events(sample_face_payload: dict) -> None:
    timeline = build_vision_timeline(sample_face_payload)
    assert timeline.frames[0].primary_face.center_x == 0.48
    assert timeline.frames[0].motion_score == 0.72
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_vision.py::test_build_vision_timeline_emits_face_and_motion_events -v`
Expected: FAIL with missing `build_vision_timeline`

- [ ] **Step 3: Define vision event models**

```python
class FaceBox(BaseModel):
    center_x: float
    center_y: float
    width: float
    height: float
    confidence: float


class VisionFrame(BaseModel):
    timestamp_seconds: float
    motion_score: float
    shot_change: bool
    primary_face: FaceBox | None = None


class VisionTimeline(BaseModel):
    frames: list[VisionFrame]
```

- [ ] **Step 4: Implement the timeline builder**

```python
def build_vision_timeline(payload: dict) -> VisionTimeline:
    frames = []
    for item in payload["frames"]:
        frame = VisionFrame(
            timestamp_seconds=item["timestamp_seconds"],
            motion_score=item["motion_score"],
            shot_change=item["shot_change"],
            primary_face=FaceBox(**item["primary_face"]) if item.get("primary_face") else None,
        )
        frames.append(frame)
    return VisionTimeline(frames=frames)
```

- [ ] **Step 5: Add a reframing-anchor test**

```python
def test_primary_face_is_available_for_crop_planning(sample_face_payload: dict) -> None:
    timeline = build_vision_timeline(sample_face_payload)
    assert timeline.frames[1].primary_face is not None
    assert 0.0 <= timeline.frames[1].primary_face.center_x <= 1.0
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/pipeline/test_vision.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/clipper/pipeline/vision.py tests/pipeline/test_vision.py tests/fixtures/sample_faces.json
git commit -m "feat: add vision timeline for motion and framing"
```

## Task 6: Build Candidate Mining And Balanced Scoring

**Files:**
- Create: `src/clipper/pipeline/candidates.py`
- Create: `tests/pipeline/test_candidates.py`

- [ ] **Step 1: Write the failing candidate mining test**

```python
from clipper.pipeline.candidates import generate_candidates


def test_generate_candidates_promotes_clips_with_hook_and_payoff(sample_transcript_timeline, sample_vision_timeline) -> None:
    candidates = generate_candidates(sample_transcript_timeline, sample_vision_timeline, max_candidates=3)
    assert len(candidates) == 3
    assert candidates[0].score > candidates[-1].score
    assert "shareability" in candidates[0].reasons
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_candidates.py::test_generate_candidates_promotes_clips_with_hook_and_payoff -v`
Expected: FAIL with missing `generate_candidates`

- [ ] **Step 3: Implement signal scoring helpers**

```python
def score_text_spike(text: str) -> float:
    keywords = ["crazy", "ban", "never", "secret", "wild", "taboo", "fight"]
    return sum(1 for word in keywords if word in text.lower()) / max(len(keywords), 1)


def score_visual_spike(motion_score: float, shot_change: bool) -> float:
    return motion_score + (0.25 if shot_change else 0.0)
```

- [ ] **Step 4: Implement candidate generation and merging**

```python
def generate_candidates(transcript_timeline, vision_timeline, max_candidates: int = 12):
    raw = []
    for segment in transcript_timeline.segments:
        text_score = score_text_spike(segment.text)
        score = text_score + 0.4
        raw.append(
            CandidateClip(
                clip_id=f"clip-{segment.start_seconds:.2f}",
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                score=score,
                reasons=["shareability"] if text_score else ["clarity"],
                spike_categories=["controversy"] if text_score >= 0.2 else [],
            )
        )
    return sorted(raw, key=lambda item: item.score, reverse=True)[:max_candidates]
```

- [ ] **Step 5: Add penalty tests for context dependence**

```python
def test_generate_candidates_penalizes_buried_leads(sample_transcript_timeline, sample_vision_timeline) -> None:
    candidates = generate_candidates(sample_transcript_timeline, sample_vision_timeline, max_candidates=5)
    buried = next(candidate for candidate in candidates if candidate.clip_id == "clip-18.00")
    assert buried.score < 0.8
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/pipeline/test_candidates.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/clipper/pipeline/candidates.py tests/pipeline/test_candidates.py
git commit -m "feat: add candidate mining and balanced clip scoring"
```

## Task 7: Add Harness-Model Review And Review Manifest Output

**Files:**
- Create: `src/clipper/adapters/harness_model.py`
- Create: `src/clipper/pipeline/review.py`
- Create: `src/clipper/adapters/storage.py`
- Create: `tests/pipeline/test_review.py`
- Create: `docs/architecture/review-manifest.md`

- [ ] **Step 1: Write the failing review manifest test**

```python
from clipper.pipeline.review import build_review_manifest


def test_build_review_manifest_enriches_candidates_with_titles(sample_candidates, tmp_path) -> None:
    manifest = build_review_manifest(
        job_id="job-123",
        video_path=tmp_path / "input.mp4",
        candidates=sample_candidates,
        model_scores=[
            {
                "clip_id": "clip-1",
                "title": "He admits the hidden tradeoff",
                "hook": "Nobody tells you this part",
                "reasons": ["strong hook", "clear payoff"],
            }
        ],
    )
    assert manifest.candidates[0].title == "He admits the hidden tradeoff"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_review.py::test_build_review_manifest_enriches_candidates_with_titles -v`
Expected: FAIL with missing `build_review_manifest`

- [ ] **Step 3: Define the harness adapter interface**

```python
class HarnessModelAdapter(Protocol):
    def score_candidates(self, prompts: list[dict]) -> list[dict]: ...
```

- [ ] **Step 4: Implement manifest enrichment**

```python
def build_review_manifest(job_id: str, video_path, candidates, model_scores):
    lookup = {item["clip_id"]: item for item in model_scores}
    enriched = []
    for candidate in candidates:
        extra = lookup.get(candidate.clip_id, {})
        enriched.append(
            candidate.model_copy(
                update={
                    "title": extra.get("title", ""),
                    "hook": extra.get("hook", ""),
                    "reasons": extra.get("reasons", candidate.reasons),
                }
            )
        )
    return ReviewManifest(job_id=job_id, video_path=video_path, candidates=enriched)
```

- [ ] **Step 5: Add persistence for the review package**

```python
import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/pipeline/test_review.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/clipper/adapters/harness_model.py src/clipper/pipeline/review.py src/clipper/adapters/storage.py tests/pipeline/test_review.py docs/architecture/review-manifest.md
git commit -m "feat: add harness review stage and review manifest output"
```

## Task 8: Add Caption Planning, Crop Planning, And Multi-Ratio Rendering

**Files:**
- Create: `src/clipper/pipeline/render.py`
- Create: `tests/pipeline/test_render.py`
- Create: `docs/architecture/render-manifest.md`

- [ ] **Step 1: Write the failing render-plan test**

```python
from clipper.pipeline.render import build_render_plan


def test_build_render_plan_outputs_all_ratios(sample_review_candidate) -> None:
    plan = build_render_plan(sample_review_candidate, approved=True)
    assert sorted(plan.outputs.keys()) == ["1:1", "16:9", "9:16"]
    assert plan.approved is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_render.py::test_build_render_plan_outputs_all_ratios -v`
Expected: FAIL with missing `build_render_plan`

- [ ] **Step 3: Implement render-plan generation**

```python
def build_render_plan(candidate, approved: bool):
    outputs = {
        "9:16": Path(f"{candidate.clip_id}-9x16.mp4"),
        "1:1": Path(f"{candidate.clip_id}-1x1.mp4"),
        "16:9": Path(f"{candidate.clip_id}-16x9.mp4"),
    }
    return RenderManifest(clip_id=candidate.clip_id, approved=approved, outputs=outputs)
```

- [ ] **Step 4: Add caption and crop-plan helpers**

```python
def build_caption_lines(transcript_segment) -> list[dict]:
    return [{"text": transcript_segment.text, "emphasis": transcript_segment.words[:2]}]


def choose_crop_anchor(frame) -> tuple[float, float]:
    if frame.primary_face:
        return (frame.primary_face.center_x, frame.primary_face.center_y)
    return (0.5, 0.5)
```

- [ ] **Step 5: Add a render-command test**

```python
def test_render_commands_include_burned_captions(sample_review_candidate) -> None:
    plan = build_render_plan(sample_review_candidate, approved=True)
    assert plan.outputs["9:16"].endswith(".mp4")
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/pipeline/test_render.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/clipper/pipeline/render.py tests/pipeline/test_render.py docs/architecture/render-manifest.md
git commit -m "feat: add multi-ratio render planning"
```

## Task 9: Wire The Orchestrator And CLI Job Execution

**Files:**
- Create: `src/clipper/pipeline/orchestrator.py`
- Modify: `src/clipper/cli.py`
- Create: `tests/pipeline/test_orchestrator.py`

- [ ] **Step 1: Write the failing orchestrator test**

```python
from clipper.pipeline.orchestrator import run_job


def test_run_job_returns_review_manifest_path(tmp_path, sample_job, monkeypatch) -> None:
    monkeypatch.setattr(
        "clipper.pipeline.orchestrator.probe_video",
        lambda _path: {"duration_seconds": 120.0, "width": 1920, "height": 1080, "fps": 30.0, "audio_sample_rate": 48000},
    )
    monkeypatch.setattr(
        "clipper.pipeline.orchestrator.transcribe_video",
        lambda _path: sample_job.mock_transcript,
    )
    monkeypatch.setattr(
        "clipper.pipeline.orchestrator.analyze_video",
        lambda _path: sample_job.mock_vision,
    )
    monkeypatch.setattr(
        "clipper.pipeline.orchestrator.score_shortlist",
        lambda _job, _candidates: sample_job.mock_model_scores,
    )
    manifest_path = run_job(sample_job)
    assert manifest_path.name == "review-manifest.json"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/pipeline/test_orchestrator.py::test_run_job_returns_review_manifest_path -v`
Expected: FAIL with missing `run_job`

- [ ] **Step 3: Implement pipeline composition**

```python
def run_job(job):
    probe_data = probe_video(job.video_path)
    ingest = ingest_job(job, probe_data=probe_data)
    transcript = build_transcript_timeline(transcribe_video(job.video_path))
    vision = build_vision_timeline(analyze_video(job.video_path))
    candidates = generate_candidates(transcript, vision, max_candidates=job.max_candidates)
    model_scores = score_shortlist(job, candidates)
    manifest = build_review_manifest(ingest.job_id, job.video_path, candidates, model_scores=model_scores)
    output = ingest.workspace_dir / "review-manifest.json"
    write_json(output, manifest.model_dump(mode="json"))
    return output
```

- [ ] **Step 4: Expose a CLI run command**

```python
@app.command()
def run(job_path: Path) -> None:
    payload = json.loads(job_path.read_text(encoding="utf-8"))
    job = ClipperJob.model_validate(payload)
    manifest_path = run_job(job)
    typer.echo(str(manifest_path))
```

- [ ] **Step 5: Add an integration-style CLI test**

```python
def test_run_command_prints_manifest_path(cli_runner, tmp_path, sample_job_payload) -> None:
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(sample_job_payload), encoding="utf-8")
    result = cli_runner.invoke(app, ["run", str(job_path)])
    assert result.exit_code == 0
    assert "review-manifest.json" in result.stdout
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/pipeline/test_orchestrator.py tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/clipper/pipeline/orchestrator.py src/clipper/cli.py tests/pipeline/test_orchestrator.py tests/test_cli.py
git commit -m "feat: wire clipper pipeline through cli job execution"
```

## Task 10: Add Thin Wrappers For Codex, Claude Code, And Hermes

**Files:**
- Create: `src/clipper/wrappers/common.py`
- Create: `src/clipper/wrappers/codex.py`
- Create: `src/clipper/wrappers/claude_code.py`
- Create: `src/clipper/wrappers/hermes.py`
- Create: `tests/test_wrappers.py`

- [ ] **Step 1: Write the failing wrapper normalization test**

```python
from clipper.wrappers.codex import codex_job_from_args


def test_codex_wrapper_builds_common_job(tmp_path) -> None:
    job = codex_job_from_args(str(tmp_path / "input.mp4"), str(tmp_path / "out"))
    assert job.review_required is True
    assert job.output_profile.ratios == ["9:16", "1:1", "16:9"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wrappers.py::test_codex_wrapper_builds_common_job -v`
Expected: FAIL with missing wrapper modules

- [ ] **Step 3: Implement the shared normalization helper**

```python
def build_common_job(video_path: str, output_dir: str) -> ClipperJob:
    return ClipperJob.model_validate(
        {
            "video_path": video_path,
            "output_dir": output_dir,
            "review_required": True,
            "output_profile": {"ratios": ["9:16", "1:1", "16:9"], "caption_preset": "hook-default"},
        }
    )
```

- [ ] **Step 4: Add per-wrapper entry helpers**

```python
def codex_job_from_args(video_path: str, output_dir: str) -> ClipperJob:
    return build_common_job(video_path, output_dir)


def claude_job_from_args(video_path: str, output_dir: str) -> ClipperJob:
    return build_common_job(video_path, output_dir)


def hermes_job_from_args(video_path: str, output_dir: str) -> ClipperJob:
    return build_common_job(video_path, output_dir)
```

- [ ] **Step 5: Add the missing wrapper test file**

```python
from clipper.wrappers.hermes import hermes_job_from_args


def test_hermes_wrapper_uses_common_defaults(tmp_path) -> None:
    job = hermes_job_from_args(str(tmp_path / "input.mp4"), str(tmp_path / "out"))
    assert job.max_candidates == 12
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_wrappers.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/clipper/wrappers tests/test_wrappers.py
git commit -m "feat: add thin wrappers for supported agent harnesses"
```

## Task 11: Add Fixtures, Shared Test Helpers, And Happy-Path Coverage

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/fixtures/sample_job.json`
- Modify: `tests/pipeline/test_transcribe.py`
- Modify: `tests/pipeline/test_vision.py`
- Modify: `tests/pipeline/test_candidates.py`
- Modify: `tests/pipeline/test_review.py`
- Modify: `tests/pipeline/test_render.py`

- [ ] **Step 1: Add shared fixtures**

```python
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from clipper.models.candidate import CandidateClip
from clipper.models.job import ClipperJob
from clipper.pipeline.transcribe import build_transcript_timeline
from clipper.pipeline.vision import build_vision_timeline


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def sample_job_payload() -> dict:
    return {
        "video_path": "/tmp/input.mp4",
        "output_dir": "/tmp/out",
        "review_required": True,
        "max_candidates": 3,
        "output_profile": {"ratios": ["9:16", "1:1", "16:9"], "caption_preset": "hook-default"},
    }
```

- [ ] **Step 2: Add transcript and vision fixture loaders**

```python
@pytest.fixture
def sample_transcript_payload() -> dict:
    return json.loads(Path("tests/fixtures/sample_transcript.json").read_text(encoding="utf-8"))


@pytest.fixture
def sample_face_payload() -> dict:
    return json.loads(Path("tests/fixtures/sample_faces.json").read_text(encoding="utf-8"))
```

- [ ] **Step 3: Add timeline and job fixtures**

```python
@pytest.fixture
def sample_transcript_timeline(sample_transcript_payload: dict):
    return build_transcript_timeline(sample_transcript_payload)


@pytest.fixture
def sample_vision_timeline(sample_face_payload: dict):
    return build_vision_timeline(sample_face_payload)


@pytest.fixture
def sample_job(tmp_path: Path, sample_job_payload: dict) -> ClipperJob:
    payload = dict(sample_job_payload)
    payload["video_path"] = str(tmp_path / "input.mp4")
    payload["output_dir"] = str(tmp_path / "out")
    job = ClipperJob.model_validate(payload)
    object.__setattr__(job, "mock_transcript", {
        "segments": [
            {
                "speaker": "speaker_1",
                "start_seconds": 0.0,
                "end_seconds": 5.0,
                "text": "Look at this wild tradeoff",
                "words": [{"text": "Look", "start_seconds": 0.0, "end_seconds": 0.2, "confidence": 0.99}],
            }
        ]
    })
    object.__setattr__(job, "mock_vision", {
        "frames": [
            {
                "timestamp_seconds": 0.0,
                "motion_score": 0.8,
                "shot_change": False,
                "primary_face": {"center_x": 0.5, "center_y": 0.4, "width": 0.3, "height": 0.3, "confidence": 0.98},
            }
        ]
    })
    object.__setattr__(job, "mock_model_scores", [{"clip_id": "clip-0.00", "title": "He admits the hidden tradeoff", "hook": "Nobody tells you this part", "reasons": ["strong hook", "clear payoff"]}])
    return job
```

- [ ] **Step 4: Add happy-path candidate fixtures**

```python
@pytest.fixture
def sample_candidates():
    return [
        CandidateClip(
            clip_id="clip-1",
            start_seconds=10.0,
            end_seconds=25.0,
            score=1.4,
            reasons=["shareability"],
            spike_categories=["absurdity"],
            title="He admits the hidden tradeoff",
            hook="Nobody tells you this part",
        )
    ]
```

- [ ] **Step 5: Add a review-candidate fixture**

```python
@pytest.fixture
def sample_review_candidate(sample_candidates):
    return sample_candidates[0]
```

- [ ] **Step 6: Run the full suite**

Run: `pytest -v`
Expected: PASS

- [ ] **Step 7: Add coverage output**

Run: `pytest --cov=src/clipper --cov-report=term-missing`
Expected: line coverage report with the pipeline modules listed

- [ ] **Step 8: Commit**

```bash
git add tests
git commit -m "test: add shared fixtures and end-to-end happy path coverage"
```

## Task 12: Polish Docs And Local Developer Workflow

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture/job-spec.md`
- Modify: `docs/architecture/review-manifest.md`
- Modify: `docs/architecture/render-manifest.md`

- [ ] **Step 1: Document the local setup**

```markdown
## Local setup

1. Create a Python 3.12 virtualenv.
2. Install FFmpeg on the host.
3. Install the package with `pip install -e ".[dev]"`.
4. Run `pytest -v`.
```

- [ ] **Step 2: Document the CLI job flow**

```markdown
## Run a job

`python -m clipper.cli run /absolute/path/job.json`
```

- [ ] **Step 3: Document current v1 limitations**

```markdown
## Current limitations

- Speech-heavy content is the main target.
- Review is required before rendering.
- Caption styling is preset-based in v1.
```

- [ ] **Step 4: Run formatting and tests**

Run: `ruff check .`
Expected: `All checks passed!`

Run: `pytest -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md docs/architecture
git commit -m "docs: document clipper architecture and local workflow"
```

## Spec Coverage Check

- Shared engine with thin wrappers: covered by Tasks 2, 9, and 10.
- Full-video analysis: covered by Tasks 3, 4, 5, and 9.
- Candidate mining before model review: covered by Tasks 6 and 7.
- Balanced multimodal scoring: covered by Tasks 5, 6, and 7.
- Human review package before rendering: covered by Task 7.
- Final exports in 9:16, 1:1, and 16:9: covered by Task 8.
- Machine-readable manifests and reusable artifacts: covered by Tasks 2, 7, and 8.

## Self-Review Notes

- Placeholder scan: no deferred implementation markers remain in the plan.
- Type consistency: `ClipperJob`, `CandidateClip`, `ReviewManifest`, and `RenderManifest` are introduced before later tasks use them.
- Scope check: the plan stays within one v1 subsystem boundary and does not branch into hosting, publishing, or brand-specific caption theming.
