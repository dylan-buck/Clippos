from __future__ import annotations

from pathlib import Path

from clippos.models.analysis import MediaProbe
from clippos.models.candidate import CandidateClip
from clippos.models.media import AspectRatio
from clippos.models.render import DEFAULT_CAPTION_PRESET, CaptionPreset, RenderManifest
from clippos.pipeline.captions import build_caption_plan
from clippos.pipeline.crops import DEFAULT_RATIOS, build_crop_plans
from clippos.pipeline.scene_strategy import derive_clip_mode
from clippos.pipeline.transcribe import TranscriptTimeline
from clippos.pipeline.vision import VisionTimeline

RENDERS_DIRNAME = "renders"
RENDER_MANIFEST_FILENAME = "render-manifest.json"


def renders_root(workspace_dir: Path) -> Path:
    return workspace_dir / RENDERS_DIRNAME


def clip_render_dir(workspace_dir: Path, clip_id: str) -> Path:
    return renders_root(workspace_dir) / clip_id


def output_video_path(workspace_dir: Path, clip_id: str, ratio: AspectRatio) -> Path:
    return clip_render_dir(workspace_dir, clip_id) / _output_filename(clip_id, ratio)


def render_manifest_path(workspace_dir: Path, clip_id: str) -> Path:
    return clip_render_dir(workspace_dir, clip_id) / RENDER_MANIFEST_FILENAME


def build_render_plan(
    *,
    candidate: CandidateClip,
    source_video: Path,
    transcript: TranscriptTimeline,
    vision: VisionTimeline,
    probe: MediaProbe,
    workspace_dir: Path,
    ratios: tuple[AspectRatio, ...] = DEFAULT_RATIOS,
    approved: bool | None = None,
    caption_preset: CaptionPreset = DEFAULT_CAPTION_PRESET,
) -> RenderManifest:
    if not ratios:
        raise ValueError("ratios must not be empty")

    caption_plan = build_caption_plan(
        transcript,
        start_seconds=candidate.start_seconds,
        end_seconds=candidate.end_seconds,
    )
    crop_plans = build_crop_plans(
        vision,
        start_seconds=candidate.start_seconds,
        end_seconds=candidate.end_seconds,
        source_width=probe.width,
        source_height=probe.height,
        ratios=ratios,
    )
    outputs = {
        ratio: output_video_path(workspace_dir, candidate.clip_id, ratio)
        for ratio in ratios
    }
    mode = derive_clip_mode(
        vision,
        start_seconds=candidate.start_seconds,
        end_seconds=candidate.end_seconds,
    )
    return RenderManifest(
        clip_id=candidate.clip_id,
        approved=candidate.approved if approved is None else approved,
        source_video=source_video,
        start_seconds=candidate.start_seconds,
        end_seconds=candidate.end_seconds,
        outputs=outputs,
        crop_plans=crop_plans,
        caption_plan=list(caption_plan),
        mode=mode,
        caption_preset=caption_preset,
    )


def _output_filename(clip_id: str, ratio: AspectRatio) -> str:
    return f"{clip_id}-{ratio.replace(':', 'x')}.mp4"
