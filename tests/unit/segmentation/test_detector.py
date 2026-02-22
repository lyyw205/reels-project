"""Tests for reels.segmentation.detector module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reels.exceptions import SegmentationError
from reels.segmentation.detector import ShotDetector


class TestShotDetector:
    def test_init_default_threshold(self) -> None:
        d = ShotDetector()
        assert d.threshold == 27.0

    def test_init_custom_threshold(self) -> None:
        d = ShotDetector(threshold=30.0)
        assert d.threshold == 30.0

    def test_file_not_found_raises(self) -> None:
        d = ShotDetector()
        with pytest.raises(SegmentationError, match="not found"):
            d.detect(Path("/nonexistent/video.mp4"))

    @pytest.mark.slow
    def test_detect_real_video(self, synthetic_video: Path | None) -> None:
        if synthetic_video is None:
            pytest.skip("ffmpeg not available")

        d = ShotDetector(threshold=20.0)
        boundaries = d.detect(synthetic_video)

        # Synthetic video has 3 color segments, so should detect at least 1 boundary
        assert len(boundaries) >= 1
        assert boundaries[0].frame_number == 0
