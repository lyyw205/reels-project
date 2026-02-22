"""Video metadata extraction via ffprobe."""

from __future__ import annotations

import logging
from pathlib import Path

from reels.exceptions import IngestError
from reels.models.metadata import VideoMetadata
from reels.utils.video import run_ffprobe

logger = logging.getLogger(__name__)


class VideoProber:
    """Extract video metadata using ffprobe."""

    def probe(self, video_path: Path) -> VideoMetadata:
        """Extract metadata from a video file.

        Returns:
            VideoMetadata with duration, fps, resolution, etc.

        Raises:
            IngestError: If the file cannot be probed or has no video stream.
        """
        data = run_ffprobe(video_path)

        video_stream = self._find_stream(data, "video")
        if video_stream is None:
            raise IngestError(f"No video stream found in {video_path}")

        fmt = data.get("format", {})
        audio_stream = self._find_stream(data, "audio")

        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        duration = float(fmt.get("duration", video_stream.get("duration", 0)))
        fps = self._parse_fps(video_stream.get("r_frame_rate", "30/1"))
        bitrate = float(fmt.get("bit_rate", 0)) / 1000 if fmt.get("bit_rate") else None
        codec = video_stream.get("codec_name")

        return VideoMetadata(
            source=str(video_path),
            duration_sec=duration,
            fps=fps,
            width=width,
            height=height,
            resolution=f"{width}x{height}",
            bitrate_kbps=bitrate,
            has_audio=audio_stream is not None,
            codec=codec,
        )

    def has_audio(self, video_path: Path) -> bool:
        """Check if a video file contains an audio stream."""
        data = run_ffprobe(video_path)
        return self._find_stream(data, "audio") is not None

    @staticmethod
    def _find_stream(data: dict, codec_type: str) -> dict | None:
        """Find the first stream of the given type."""
        for stream in data.get("streams", []):
            if stream.get("codec_type") == codec_type:
                return stream
        return None

    @staticmethod
    def _parse_fps(r_frame_rate: str) -> float:
        """Parse fps from ffprobe r_frame_rate (e.g., '30/1', '30000/1001')."""
        try:
            parts = r_frame_rate.split("/")
            if len(parts) == 2:
                num, den = float(parts[0]), float(parts[1])
                return round(num / den, 2) if den > 0 else 30.0
            return float(parts[0])
        except (ValueError, ZeroDivisionError):
            return 30.0
