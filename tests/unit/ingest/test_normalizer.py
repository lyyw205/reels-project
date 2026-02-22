"""Tests for reels.ingest.normalizer module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reels.exceptions import NormalizationError
from reels.ingest.normalizer import VideoNormalizer
from reels.models.metadata import NormalizeResult, VideoMetadata


DEFAULT_CONFIG = {
    "ingest": {
        "normalize": {
            "strategy": "preserve_aspect",
            "max_short_side": 1080,
            "target_fps": 30,
            "audio_sample_rate": 16000,
            "video_codec": "libx264",
            "audio_codec": "aac",
        }
    }
}


class TestVideoNormalizerInit:
    def test_defaults_from_config(self) -> None:
        n = VideoNormalizer(DEFAULT_CONFIG)
        assert n.strategy == "preserve_aspect"
        assert n.max_short_side == 1080
        assert n.target_fps == 30

    def test_empty_config_uses_defaults(self) -> None:
        n = VideoNormalizer({})
        assert n.strategy == "preserve_aspect"
        assert n.max_short_side == 1080


class TestBuildVideoFilter:
    def test_portrait_small_enough(self) -> None:
        n = VideoNormalizer(DEFAULT_CONFIG)
        vf = n._build_video_filter(720, 1280)
        assert "trunc" in vf  # Already small, just ensure even dims

    def test_portrait_needs_scale(self) -> None:
        n = VideoNormalizer(DEFAULT_CONFIG)
        vf = n._build_video_filter(2160, 3840)
        assert "1080" in vf
        assert "scale=" in vf

    def test_landscape(self) -> None:
        n = VideoNormalizer(DEFAULT_CONFIG)
        vf = n._build_video_filter(3840, 2160)
        assert "1080" in vf

    def test_force_portrait_strategy(self) -> None:
        config = {"ingest": {"normalize": {"strategy": "force_portrait"}}}
        n = VideoNormalizer(config)
        vf = n._build_video_filter(1920, 1080)
        assert "pad=" in vf


class TestVideoNormalizerNormalize:
    def test_input_not_found_raises(self, tmp_path: Path) -> None:
        n = VideoNormalizer(DEFAULT_CONFIG)
        with pytest.raises(NormalizationError, match="not found"):
            n.normalize(tmp_path / "nonexistent.mp4", tmp_path / "out")

    @pytest.mark.slow
    def test_normalize_real_video(self, synthetic_video: Path | None, tmp_path: Path) -> None:
        if synthetic_video is None:
            pytest.skip("ffmpeg not available")

        n = VideoNormalizer(DEFAULT_CONFIG)
        result = n.normalize(synthetic_video, tmp_path / "normalized")

        assert isinstance(result, NormalizeResult)
        assert Path(result.video_path).exists()
        assert result.metadata.fps == 30.0
