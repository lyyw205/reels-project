"""Video normalization via ffmpeg."""

from __future__ import annotations

import logging
from pathlib import Path

from reels.exceptions import NormalizationError
from reels.models.metadata import NormalizeResult
from reels.ingest.probe import VideoProber
from reels.utils.audio import extract_audio_track
from reels.utils.video import run_ffmpeg

logger = logging.getLogger(__name__)


class VideoNormalizer:
    """Normalize video to standard format using ffmpeg.

    Supports two strategies:
    - preserve_aspect: Keep original aspect ratio, scale short side to max_short_side.
    - force_portrait: Force 9:16 portrait (pad if necessary).
    """

    def __init__(self, config: dict) -> None:
        normalize_cfg = config.get("ingest", {}).get("normalize", {})
        self.strategy: str = normalize_cfg.get("strategy", "preserve_aspect")
        self.max_short_side: int = normalize_cfg.get("max_short_side", 1080)
        self.target_fps: int = normalize_cfg.get("target_fps", 30)
        self.audio_sample_rate: int = normalize_cfg.get("audio_sample_rate", 16000)
        self.video_codec: str = normalize_cfg.get("video_codec", "libx264")
        self.audio_codec: str = normalize_cfg.get("audio_codec", "aac")
        self._prober = VideoProber()

    def normalize(self, input_path: Path, output_dir: Path) -> NormalizeResult:
        """Normalize video file.

        Args:
            input_path: Path to source video.
            output_dir: Directory for output files.

        Returns:
            NormalizeResult with paths and metadata.

        Raises:
            NormalizationError: If normalization fails.
        """
        if not input_path.exists():
            raise NormalizationError(f"Input video not found: {input_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        video_out = output_dir / f"{input_path.stem}_normalized.mp4"

        # Probe original to determine scaling
        original_meta = self._prober.probe(input_path)
        vf = self._build_video_filter(original_meta.width, original_meta.height)

        args = [
            "-i", str(input_path),
            "-vf", vf,
            "-r", str(self.target_fps),
            "-c:v", self.video_codec,
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
        ]

        if original_meta.has_audio:
            args.extend(["-c:a", self.audio_codec, "-ar", str(self.audio_sample_rate)])
        else:
            args.extend(["-an"])

        args.append(str(video_out))
        run_ffmpeg(args)

        if not video_out.exists():
            raise NormalizationError(f"Normalization produced no output: {video_out}")

        # Extract audio separately for analysis (WAV, mono, 16kHz)
        audio_out: Path | None = None
        if original_meta.has_audio:
            audio_out = output_dir / f"{input_path.stem}_audio.wav"
            extract_audio_track(input_path, audio_out, sample_rate=self.audio_sample_rate)

        # Probe normalized video for final metadata
        metadata = self._prober.probe(video_out)

        logger.info(
            "Normalized: %s → %s (%dx%d, %.1fs)",
            input_path.name, video_out.name,
            metadata.width, metadata.height, metadata.duration_sec,
        )

        return NormalizeResult(
            video_path=str(video_out),
            audio_path=str(audio_out) if audio_out else None,
            metadata=metadata,
        )

    def _build_video_filter(self, width: int, height: int) -> str:
        """Build ffmpeg video filter string based on strategy."""
        if self.strategy == "force_portrait":
            return (
                f"scale={self.max_short_side}:{self.max_short_side * 16 // 9}"
                f":force_original_aspect_ratio=decrease,"
                f"pad={self.max_short_side}:{self.max_short_side * 16 // 9}:(ow-iw)/2:(oh-ih)/2"
            )

        # preserve_aspect: scale so the shorter side = max_short_side
        short_side = min(width, height)
        if short_side <= self.max_short_side:
            # Already small enough, just ensure even dimensions
            return "scale=trunc(iw/2)*2:trunc(ih/2)*2"

        if width < height:
            # Portrait: width is short side
            return f"scale={self.max_short_side}:-2"
        else:
            # Landscape: height is short side
            return f"scale=-2:{self.max_short_side}"
