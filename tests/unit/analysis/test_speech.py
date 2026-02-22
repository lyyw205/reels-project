"""Tests for reels.analysis.speech module."""

from __future__ import annotations

from pathlib import Path

import pytest

from reels.analysis.base import AnalysisContext
from reels.analysis.speech import SpeechAnalyzer
from reels.models.analysis import SpeechResult
from reels.models.metadata import VideoMetadata
from reels.models.shot import Shot


MOCK_META = VideoMetadata(
    source="test.mp4", duration_sec=10.0, fps=30.0,
    width=1080, height=1920, resolution="1080x1920",
)


class TestSpeechAnalyzer:
    def test_init_defaults(self) -> None:
        a = SpeechAnalyzer()
        assert a.name == "speech"
        assert a.model_size == "small"
        assert a.compute_type == "int8"
        assert a.word_timestamps is True

    def test_init_from_config(self) -> None:
        cfg = {"analysis": {"speech": {"model": "large-v3", "language": "en"}}}
        a = SpeechAnalyzer(cfg)
        assert a.model_size == "large-v3"
        assert a.language == "en"

    def test_no_audio_returns_no_speech(self) -> None:
        a = SpeechAnalyzer()
        ctx = AnalysisContext(
            video_path=Path("/test.mp4"), audio_path=None,
            work_dir=Path("/tmp"), metadata=MOCK_META,
        )
        shot = Shot(shot_id=0, start_sec=0.0, end_sec=3.0, start_frame=0, end_frame=90, duration_sec=3.0)
        result = a.analyze_shot(shot, ctx)
        assert isinstance(result, SpeechResult)
        assert result.has_speech is False

    def test_missing_audio_file_returns_no_speech(self) -> None:
        a = SpeechAnalyzer()
        ctx = AnalysisContext(
            video_path=Path("/test.mp4"),
            audio_path=Path("/nonexistent/audio.wav"),
            work_dir=Path("/tmp"), metadata=MOCK_META,
        )
        shot = Shot(shot_id=0, start_sec=0.0, end_sec=3.0, start_frame=0, end_frame=90, duration_sec=3.0)
        result = a.analyze_shot(shot, ctx)
        assert result.has_speech is False

    def test_cleanup(self) -> None:
        a = SpeechAnalyzer()
        a._model = "mock"
        a.cleanup()
        assert a._model is None
