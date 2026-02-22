"""Shot boundary detection using PySceneDetect."""

from __future__ import annotations

import logging
from pathlib import Path

from reels.exceptions import SegmentationError
from reels.models.shot import ShotBoundary

logger = logging.getLogger(__name__)


class ShotDetector:
    """Detect shot boundaries using PySceneDetect's ContentDetector."""

    def __init__(self, threshold: float = 27.0) -> None:
        self.threshold = threshold

    def detect(self, video_path: Path) -> list[ShotBoundary]:
        """Detect shot boundaries in a video.

        Args:
            video_path: Path to the video file.

        Returns:
            List of ShotBoundary objects at each detected cut point.

        Raises:
            SegmentationError: If detection fails.
        """
        try:
            from scenedetect import open_video, SceneManager
            from scenedetect.detectors import ContentDetector
        except ImportError as e:
            raise SegmentationError("PySceneDetect is not installed") from e

        if not video_path.exists():
            raise SegmentationError(f"Video file not found: {video_path}")

        try:
            video = open_video(str(video_path))
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=self.threshold))
            scene_manager.detect_scenes(video, show_progress=False)

            scene_list = scene_manager.get_scene_list()
            fps = video.frame_rate

            boundaries: list[ShotBoundary] = []

            # First boundary is always frame 0
            boundaries.append(ShotBoundary(frame_number=0, timecode_sec=0.0))

            for _start, end in scene_list[:-1]:
                # Each scene end (except last) is a boundary
                frame_num = end.get_frames()
                time_sec = frame_num / fps if fps > 0 else 0.0
                boundaries.append(ShotBoundary(
                    frame_number=frame_num,
                    timecode_sec=round(time_sec, 3),
                ))

            logger.info(
                "Detected %d boundaries in %s (threshold=%.1f)",
                len(boundaries), video_path.name, self.threshold,
            )
            return boundaries

        except SegmentationError:
            raise
        except Exception as e:
            raise SegmentationError(f"Shot detection failed for {video_path}: {e}") from e
