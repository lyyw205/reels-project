"""Ingest module: download, normalize, and extract metadata from videos."""

from __future__ import annotations

import logging
from pathlib import Path

from reels.exceptions import DurationLimitError
from reels.ingest.downloader import VideoDownloader
from reels.ingest.normalizer import VideoNormalizer
from reels.ingest.probe import VideoProber
from reels.models.metadata import IngestResult

logger = logging.getLogger(__name__)


def ingest_video(
    source: str,
    work_dir: Path,
    config: dict,
) -> IngestResult:
    """Run the full ingest pipeline: download -> normalize -> extract metadata.

    Args:
        source: URL or local file path.
        work_dir: Working directory for output files.
        config: Pipeline configuration dict.

    Returns:
        IngestResult with original path, normalized video, audio, and metadata.

    Raises:
        DurationLimitError: If video exceeds max_video_duration_sec.
        IngestError: For other ingest failures.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    download_dir = work_dir / "downloads"
    normalize_dir = work_dir / "normalized"

    max_duration = config.get("pipeline", {}).get("max_video_duration_sec", 300)

    # Step 1: Download (or resolve local path)
    downloader = VideoDownloader(download_dir, config)
    original_path = downloader.download(source)
    logger.info("Source resolved: %s", original_path)

    # Step 2: Check duration limit
    prober = VideoProber()
    original_meta = prober.probe(original_path)

    if original_meta.duration_sec > max_duration:
        raise DurationLimitError(
            f"Video duration {original_meta.duration_sec:.1f}s exceeds "
            f"limit of {max_duration}s: {original_path}"
        )

    # Step 3: Normalize
    normalizer = VideoNormalizer(config)
    result = normalizer.normalize(original_path, normalize_dir)

    logger.info(
        "Ingest complete: %s → %s (%.1fs, %dx%d)",
        original_path.name,
        Path(result.video_path).name,
        result.metadata.duration_sec,
        result.metadata.width,
        result.metadata.height,
    )

    return IngestResult(
        original_path=str(original_path),
        normalized_video=result.video_path,
        audio_path=result.audio_path,
        metadata=result.metadata,
    )
