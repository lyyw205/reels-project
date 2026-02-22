"""Tests for reels.ingest.probe module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from reels.exceptions import IngestError
from reels.ingest.probe import VideoProber
from reels.models.metadata import VideoMetadata


SAMPLE_FFPROBE_OUTPUT = {
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1080,
            "height": 1920,
            "r_frame_rate": "30/1",
            "duration": "15.0",
        },
        {
            "codec_type": "audio",
            "codec_name": "aac",
            "sample_rate": "44100",
        },
    ],
    "format": {
        "duration": "15.0",
        "bit_rate": "2000000",
    },
}

SAMPLE_FFPROBE_NO_AUDIO = {
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "r_frame_rate": "24000/1001",
            "duration": "10.0",
        },
    ],
    "format": {
        "duration": "10.0",
    },
}


class TestVideoProber:
    def test_probe_returns_metadata(self) -> None:
        prober = VideoProber()
        with patch("reels.ingest.probe.run_ffprobe", return_value=SAMPLE_FFPROBE_OUTPUT):
            meta = prober.probe(Path("test.mp4"))

        assert isinstance(meta, VideoMetadata)
        assert meta.width == 1080
        assert meta.height == 1920
        assert meta.fps == 30.0
        assert meta.duration_sec == 15.0
        assert meta.has_audio is True
        assert meta.codec == "h264"
        assert meta.bitrate_kbps == 2000.0

    def test_probe_landscape_no_audio(self) -> None:
        prober = VideoProber()
        with patch("reels.ingest.probe.run_ffprobe", return_value=SAMPLE_FFPROBE_NO_AUDIO):
            meta = prober.probe(Path("landscape.mp4"))

        assert meta.width == 1920
        assert meta.height == 1080
        assert meta.has_audio is False
        assert meta.fps == pytest.approx(23.98, abs=0.01)

    def test_probe_no_video_stream_raises(self) -> None:
        prober = VideoProber()
        data = {"streams": [{"codec_type": "audio"}], "format": {}}
        with patch("reels.ingest.probe.run_ffprobe", return_value=data):
            with pytest.raises(IngestError, match="No video stream"):
                prober.probe(Path("audio_only.mp4"))

    def test_has_audio_true(self) -> None:
        prober = VideoProber()
        with patch("reels.ingest.probe.run_ffprobe", return_value=SAMPLE_FFPROBE_OUTPUT):
            assert prober.has_audio(Path("test.mp4")) is True

    def test_has_audio_false(self) -> None:
        prober = VideoProber()
        with patch("reels.ingest.probe.run_ffprobe", return_value=SAMPLE_FFPROBE_NO_AUDIO):
            assert prober.has_audio(Path("test.mp4")) is False

    def test_parse_fps_fraction(self) -> None:
        assert VideoProber._parse_fps("30/1") == 30.0
        assert VideoProber._parse_fps("24000/1001") == pytest.approx(23.98, abs=0.01)
        assert VideoProber._parse_fps("60") == 60.0
        assert VideoProber._parse_fps("invalid") == 30.0

    @pytest.mark.slow
    def test_probe_real_video(self, synthetic_video: Path | None) -> None:
        if synthetic_video is None:
            pytest.skip("ffmpeg not available")
        prober = VideoProber()
        meta = prober.probe(synthetic_video)
        assert meta.width == 320
        assert meta.height == 568
        assert meta.duration_sec > 0
        assert meta.fps > 0
