"""Image processing utilities: SSIM, keyframe extraction."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index between two images.

    Both images must be the same size. Converts to grayscale internally.

    Returns:
        SSIM score in [0, 1]. Higher means more similar.
    """
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(gray1.astype(np.float64), (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(gray2.astype(np.float64), (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(gray1.astype(np.float64) ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gray2.astype(np.float64) ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(gray1.astype(np.float64) * gray2.astype(np.float64), (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean())


def extract_keyframes(
    video_path: Path,
    shot_id: int,
    start_frame: int,
    end_frame: int,
    output_dir: Path,
    num_keyframes: int = 3,
    quality: int = 85,
) -> list[str]:
    """Extract evenly-spaced keyframes from a shot.

    Args:
        video_path: Path to source video.
        shot_id: Shot identifier for naming.
        start_frame: First frame of the shot.
        end_frame: Last frame of the shot (exclusive).
        output_dir: Directory to save keyframe images.
        num_keyframes: Number of keyframes to extract.
        quality: JPEG quality (1-100).

    Returns:
        List of relative paths to saved keyframe images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return []

    total_frames = end_frame - start_frame
    if total_frames <= 0:
        cap.release()
        return []

    # Distribute keyframes evenly across the shot
    if num_keyframes == 1:
        target_frames = [start_frame + total_frames // 2]
    else:
        step = total_frames / (num_keyframes + 1)
        target_frames = [start_frame + int(step * (i + 1)) for i in range(num_keyframes)]

    saved_paths: list[str] = []
    for idx, frame_num in enumerate(target_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        filename = f"shot_{shot_id:04d}_kf_{idx}.jpg"
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        saved_paths.append(filename)

    cap.release()
    logger.debug("Extracted %d keyframes for shot %d", len(saved_paths), shot_id)
    return saved_paths


def read_frame_at(video_path: Path, frame_number: int) -> np.ndarray | None:
    """Read a single frame from video at the given frame number."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None
