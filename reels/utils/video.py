"""Low-level ffmpeg/ffprobe wrappers."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from reels.exceptions import IngestError, NormalizationError

logger = logging.getLogger(__name__)


def _check_binary(name: str) -> str:
    """Check that a binary exists on PATH and return its path."""
    path = shutil.which(name)
    if path is None:
        raise IngestError(f"{name} not found on PATH. Please install ffmpeg.")
    return path


def run_ffprobe(video_path: Path) -> dict[str, Any]:
    """Run ffprobe and return parsed JSON output.

    Raises:
        IngestError: If ffprobe is not found or the file cannot be probed.
    """
    ffprobe = _check_binary("ffprobe")

    if not video_path.exists():
        raise IngestError(f"Video file not found: {video_path}")

    cmd = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        raise IngestError(f"ffprobe failed for {video_path}: {e.stderr}") from e
    except subprocess.TimeoutExpired as e:
        raise IngestError(f"ffprobe timed out for {video_path}") from e
    except json.JSONDecodeError as e:
        raise IngestError(f"ffprobe returned invalid JSON for {video_path}") from e


def run_ffmpeg(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg with given arguments.

    Raises:
        NormalizationError: If ffmpeg is not found or the command fails.
    """
    ffmpeg = _check_binary("ffmpeg")
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *args]

    logger.debug("Running ffmpeg: %s", " ".join(cmd))

    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
    except subprocess.CalledProcessError as e:
        raise NormalizationError(f"ffmpeg failed: {e.stderr.strip()}") from e
    except subprocess.TimeoutExpired as e:
        raise NormalizationError(f"ffmpeg timed out after {timeout}s") from e
