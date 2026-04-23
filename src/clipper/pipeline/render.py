from pathlib import Path

from clipper.models.render import RenderManifest


def build_render_plan(candidate, approved: bool) -> RenderManifest:
    outputs = {
        "9:16": Path(f"{candidate.clip_id}-9x16.mp4"),
        "1:1": Path(f"{candidate.clip_id}-1x1.mp4"),
        "16:9": Path(f"{candidate.clip_id}-16x9.mp4"),
    }
    return RenderManifest(
        clip_id=candidate.clip_id,
        approved=approved,
        outputs=outputs,
    )


def build_caption_lines(transcript_segment) -> list[dict]:
    return [{"text": transcript_segment.text, "emphasis": transcript_segment.words[:2]}]


def choose_crop_anchor(frame) -> tuple[float, float]:
    if frame.primary_face:
        return (frame.primary_face.center_x, frame.primary_face.center_y)
    return (0.5, 0.5)
