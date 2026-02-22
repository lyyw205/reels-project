"""Tests for reels.ingest facade (ingest_video)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reels.exceptions import DurationLimitError
from reels.ingest import ingest_video
from reels.models.metadata import IngestResult, NormalizeResult, VideoMetadata


MOCK_META = VideoMetadata(
    source="test.mp4",
    duration_sec=15.0,
    fps=30.0,
    width=1080,
    height=1920,
    resolution="1080x1920",
    has_audio=True,
)

MOCK_META_LONG = VideoMetadata(
    source="long.mp4",
    duration_sec=600.0,
    fps=30.0,
    width=1080,
    height=1920,
    resolution="1080x1920",
    has_audio=True,
)

MOCK_CONFIG = {
    "pipeline": {"max_video_duration_sec": 300},
    "ingest": {"normalize": {}},
}


class TestIngestVideo:
    @patch("reels.ingest.VideoNormalizer")
    @patch("reels.ingest.VideoProber")
    @patch("reels.ingest.VideoDownloader")
    def test_local_file_ingest(
        self,
        mock_dl_cls: MagicMock,
        mock_prober_cls: MagicMock,
        mock_norm_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        local_file = tmp_path / "test.mp4"
        local_file.write_bytes(b"\x00")

        mock_dl_cls.return_value.download.return_value = local_file
        mock_prober_cls.return_value.probe.return_value = MOCK_META
        mock_norm_cls.return_value.normalize.return_value = NormalizeResult(
            video_path=str(tmp_path / "norm.mp4"),
            audio_path=str(tmp_path / "audio.wav"),
            metadata=MOCK_META,
        )

        result = ingest_video(str(local_file), tmp_path / "work", MOCK_CONFIG)

        assert isinstance(result, IngestResult)
        assert result.metadata.duration_sec == 15.0
        mock_dl_cls.return_value.download.assert_called_once()
        mock_norm_cls.return_value.normalize.assert_called_once()

    @patch("reels.ingest.VideoNormalizer")
    @patch("reels.ingest.VideoProber")
    @patch("reels.ingest.VideoDownloader")
    def test_duration_limit_exceeded(
        self,
        mock_dl_cls: MagicMock,
        mock_prober_cls: MagicMock,
        mock_norm_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        local_file = tmp_path / "long.mp4"
        local_file.write_bytes(b"\x00")

        mock_dl_cls.return_value.download.return_value = local_file
        mock_prober_cls.return_value.probe.return_value = MOCK_META_LONG

        with pytest.raises(DurationLimitError, match="600.0s exceeds"):
            ingest_video(str(local_file), tmp_path / "work", MOCK_CONFIG)

        mock_norm_cls.return_value.normalize.assert_not_called()

    @pytest.mark.slow
    def test_real_ingest(self, synthetic_video: Path | None, tmp_path: Path) -> None:
        if synthetic_video is None:
            pytest.skip("ffmpeg not available")

        config = {
            "pipeline": {"max_video_duration_sec": 300},
            "ingest": {
                "normalize": {
                    "strategy": "preserve_aspect",
                    "max_short_side": 1080,
                    "target_fps": 30,
                    "audio_sample_rate": 16000,
                    "video_codec": "libx264",
                    "audio_codec": "aac",
                }
            },
        }

        result = ingest_video(str(synthetic_video), tmp_path / "work", config)
        assert isinstance(result, IngestResult)
        assert Path(result.normalized_video).exists()
        assert result.metadata.duration_sec > 0
