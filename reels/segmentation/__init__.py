"""Shot segmentation module: detect boundaries, post-process, extract keyframes."""

from __future__ import annotations

import logging
from pathlib import Path

from reels.exceptions import SegmentationError as SegmentationError
from reels.models.metadata import VideoMetadata
from reels.models.shot import SegmentationResult, Shot as Shot
from reels.segmentation.detector import ShotDetector
from reels.segmentation.postprocess import (
    boundaries_to_shots,
    merge_short_shots,
    merge_similar_shots,
)
from reels.utils.image import extract_keyframes

logger = logging.getLogger(__name__)


def segment_video(
    video_path: Path,
    metadata: VideoMetadata,
    work_dir: Path,
    config: dict,
) -> SegmentationResult:
    """Run full segmentation pipeline: detect -> postprocess -> extract keyframes.

    Args:
        video_path: Path to the (normalized) video.
        metadata: Video metadata from probe.
        work_dir: Working directory.
        config: Pipeline config dict.

    Returns:
        SegmentationResult with shots and keyframe directory.

    Raises:
        SegmentationError: If segmentation fails.
    """
    seg_cfg = config.get("segmentation", {})
    threshold = seg_cfg.get("threshold", 27.0)
    min_duration = seg_cfg.get("min_shot_duration_sec", 0.25)
    merge_threshold = seg_cfg.get("merge_similar_threshold", 0.92)
    keyframes_per_shot = config.get("analysis", {}).get("place", {}).get("keyframes_per_shot", 3)

    keyframe_dir = work_dir / "keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Detect boundaries
    detector = ShotDetector(threshold=threshold)
    boundaries = detector.detect(video_path)

    # Step 2: Convert to shots
    shots = boundaries_to_shots(boundaries, metadata)
    logger.info("Initial shot count: %d", len(shots))

    # Step 3: Merge short shots
    shots = merge_short_shots(shots, min_duration_sec=min_duration)
    logger.info("After short-shot merge: %d", len(shots))

    # Step 4: Merge visually similar consecutive shots
    shots = merge_similar_shots(shots, video_path, similarity_threshold=merge_threshold)
    logger.info("After similar-shot merge: %d", len(shots))

    # Step 5: Extract keyframes for each shot
    for shot in shots:
        kf_paths = extract_keyframes(
            video_path=video_path,
            shot_id=shot.shot_id,
            start_frame=shot.start_frame,
            end_frame=shot.end_frame,
            output_dir=keyframe_dir,
            num_keyframes=keyframes_per_shot,
        )
        shot.keyframe_paths = kf_paths

    logger.info(
        "Segmentation complete: %d shots, keyframes at %s",
        len(shots), keyframe_dir,
    )

    return SegmentationResult(
        shots=shots,
        keyframe_dir=str(keyframe_dir),
    )
