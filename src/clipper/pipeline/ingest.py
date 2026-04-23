from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

from clipper.adapters.ffmpeg import normalize_probe_data, probe_media
from clipper.models.analysis import MediaProbe
from clipper.models.job import ClipperJob


@dataclass(frozen=True)
class IngestResult:
    job_id: str
    workspace_dir: Path
    probe: MediaProbe


def _canonical_video_path(video_path: Path) -> Path:
    return video_path.expanduser().resolve(strict=False)


def ingest_job(job: ClipperJob, probe_data: dict | None = None) -> IngestResult:
    canonical_video_path = _canonical_video_path(job.video_path)
    job_id = sha1(str(canonical_video_path).encode()).hexdigest()[:12]
    workspace_dir = job.output_dir / "jobs" / job_id

    resolved_probe_data = (
        probe_data if probe_data is not None else probe_media(canonical_video_path)
    )
    probe = MediaProbe.model_validate(normalize_probe_data(resolved_probe_data))

    workspace_dir.mkdir(parents=True, exist_ok=True)

    return IngestResult(job_id=job_id, workspace_dir=workspace_dir, probe=probe)
