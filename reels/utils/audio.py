"""Audio extraction utilities."""

from __future__ import annotations

import logging
from pathlib import Path

from reels.exceptions import NormalizationError
from reels.utils.video import run_ffmpeg

logger = logging.getLogger(__name__)


def extract_audio_track(
    video_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    codec: str = "pcm_s16le",
) -> Path:
    """Extract audio track from video as WAV.

    Args:
        video_path: Path to source video.
        output_path: Where to write the audio file.
        sample_rate: Target sample rate (default 16kHz for whisper).
        codec: Audio codec for output.

    Returns:
        Path to the extracted audio file.

    Raises:
        NormalizationError: If extraction fails.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "-i", str(video_path),
        "-vn",
        "-acodec", codec,
        "-ar", str(sample_rate),
        "-ac", "1",
        str(output_path),
    ]

    run_ffmpeg(args)

    if not output_path.exists():
        raise NormalizationError(f"Audio extraction produced no output: {output_path}")

    logger.info("Extracted audio: %s (sr=%d)", output_path.name, sample_rate)
    return output_path
