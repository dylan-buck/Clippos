from dataclasses import dataclass
from pathlib import Path

from clippos.adapters.ffmpeg import normalize_probe_data, probe_media
from clippos.models.analysis import MediaProbe
from clippos.models.job import ClipposJob
from clippos.pipeline.fingerprint import (
    canonical_video_path as _canonical_video_path_impl,
    compute_video_fingerprint,
)


@dataclass(frozen=True)
class IngestResult:
    job_id: str
    workspace_dir: Path
    probe: MediaProbe
    source_fingerprint: str


def _canonical_video_path(video_path: Path) -> Path:
    return _canonical_video_path_impl(video_path)


def ingest_job(job: ClipposJob, probe_data: dict | None = None) -> IngestResult:
    canonical_video_path = _canonical_video_path(job.video_path)
    # The job_id IS the source fingerprint. Folding st_size + st_mtime_ns
    # + clippos.__version__ in here means re-encoding/replacing the file
    # at the same path lands in a fresh workspace, so stale
    # transcript.json / vision.json artifacts from the previous content
    # cannot leak into the new run.
    source_fingerprint = compute_video_fingerprint(canonical_video_path)
    job_id = source_fingerprint
    workspace_dir = job.output_dir / "jobs" / job_id

    resolved_probe_data = (
        probe_data if probe_data is not None else probe_media(canonical_video_path)
    )
    probe = MediaProbe.model_validate(normalize_probe_data(resolved_probe_data))

    workspace_dir.mkdir(parents=True, exist_ok=True)

    return IngestResult(
        job_id=job_id,
        workspace_dir=workspace_dir,
        probe=probe,
        source_fingerprint=source_fingerprint,
    )
