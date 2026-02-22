"""Shot post-processing: merge short shots, merge similar consecutive frames."""

from __future__ import annotations

import logging
from pathlib import Path

from reels.models.metadata import VideoMetadata
from reels.models.shot import Shot, ShotBoundary
from reels.utils.image import compute_ssim, read_frame_at

logger = logging.getLogger(__name__)


def boundaries_to_shots(
    boundaries: list[ShotBoundary],
    metadata: VideoMetadata,
) -> list[Shot]:
    """Convert shot boundaries into Shot objects.

    Args:
        boundaries: Sorted list of shot boundaries.
        metadata: Video metadata (for total duration and fps).

    Returns:
        List of Shot objects spanning the full video.
    """
    if not boundaries:
        # Entire video is a single shot
        total_frames = int(metadata.duration_sec * metadata.fps)
        return [Shot(
            shot_id=0,
            start_sec=0.0,
            end_sec=metadata.duration_sec,
            start_frame=0,
            end_frame=total_frames,
            duration_sec=metadata.duration_sec,
        )]

    shots: list[Shot] = []
    sorted_boundaries = sorted(boundaries, key=lambda b: b.frame_number)
    total_frames = int(metadata.duration_sec * metadata.fps)

    for i, boundary in enumerate(sorted_boundaries):
        start_frame = boundary.frame_number
        start_sec = boundary.timecode_sec

        if i + 1 < len(sorted_boundaries):
            end_frame = sorted_boundaries[i + 1].frame_number
            end_sec = sorted_boundaries[i + 1].timecode_sec
        else:
            end_frame = total_frames
            end_sec = metadata.duration_sec

        duration = round(end_sec - start_sec, 3)

        shots.append(Shot(
            shot_id=i,
            start_sec=round(start_sec, 3),
            end_sec=round(end_sec, 3),
            start_frame=start_frame,
            end_frame=end_frame,
            duration_sec=duration,
        ))

    return shots


def merge_short_shots(
    shots: list[Shot],
    min_duration_sec: float = 0.25,
) -> list[Shot]:
    """Merge shots shorter than min_duration_sec into adjacent shots.

    Short shots are merged into the previous shot (or next if it's the first).

    Args:
        shots: List of shots to process.
        min_duration_sec: Minimum shot duration threshold.

    Returns:
        List of merged shots with re-assigned IDs.
    """
    if not shots:
        return []

    merged: list[Shot] = []

    for shot in shots:
        if shot.duration_sec < min_duration_sec and merged:
            # Merge into previous shot
            prev = merged[-1]
            merged[-1] = Shot(
                shot_id=prev.shot_id,
                start_sec=prev.start_sec,
                end_sec=shot.end_sec,
                start_frame=prev.start_frame,
                end_frame=shot.end_frame,
                duration_sec=round(shot.end_sec - prev.start_sec, 3),
                keyframe_paths=prev.keyframe_paths,
            )
        else:
            merged.append(shot)

    # Re-assign shot IDs
    for i, shot in enumerate(merged):
        merged[i] = shot.model_copy(update={"shot_id": i})

    logger.debug(
        "Merged short shots: %d → %d (min_duration=%.2fs)",
        len(shots), len(merged), min_duration_sec,
    )
    return merged


def merge_similar_shots(
    shots: list[Shot],
    video_path: Path,
    similarity_threshold: float = 0.92,
) -> list[Shot]:
    """Merge consecutive shots with visually similar boundary frames.

    Uses SSIM to compare the last frame of one shot with the first frame of the next.

    Args:
        shots: List of shots to process.
        video_path: Path to video for frame extraction.
        similarity_threshold: SSIM threshold above which shots are merged.

    Returns:
        List of shots after merging similar consecutive pairs.
    """
    if len(shots) <= 1:
        return shots

    merged: list[Shot] = [shots[0]]

    for shot in shots[1:]:
        prev = merged[-1]

        # Read boundary frames
        prev_frame = read_frame_at(video_path, max(0, prev.end_frame - 1))
        curr_frame = read_frame_at(video_path, shot.start_frame)

        should_merge = False
        if prev_frame is not None and curr_frame is not None:
            ssim = compute_ssim(prev_frame, curr_frame)
            if ssim >= similarity_threshold:
                should_merge = True
                logger.debug(
                    "Merging shots %d+%d (SSIM=%.3f >= %.3f)",
                    prev.shot_id, shot.shot_id, ssim, similarity_threshold,
                )

        if should_merge:
            merged[-1] = Shot(
                shot_id=prev.shot_id,
                start_sec=prev.start_sec,
                end_sec=shot.end_sec,
                start_frame=prev.start_frame,
                end_frame=shot.end_frame,
                duration_sec=round(shot.end_sec - prev.start_sec, 3),
                keyframe_paths=prev.keyframe_paths,
            )
        else:
            merged.append(shot)

    # Re-assign shot IDs
    for i, shot in enumerate(merged):
        merged[i] = shot.model_copy(update={"shot_id": i})

    logger.debug(
        "Merged similar shots: %d → %d (threshold=%.2f)",
        len(shots), len(merged), similarity_threshold,
    )
    return merged
