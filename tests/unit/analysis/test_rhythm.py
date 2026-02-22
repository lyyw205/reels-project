"""Tests for reels.analysis.rhythm module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from reels.analysis.base import AnalysisContext
from reels.analysis.rhythm import RhythmAnalyzer
from reels.models.analysis import RhythmResult
from reels.models.metadata import VideoMetadata
from reels.models.shot import Shot


MOCK_META = VideoMetadata(
    source="test.mp4", duration_sec=10.0, fps=30.0,
    width=1080, height=1920, resolution="1080x1920",
)


class TestRhythmAnalyzer:
    def test_init_defaults(self) -> None:
        a = RhythmAnalyzer()
        assert a.name == "rhythm"
        assert a.hop_length == 512

    def test_no_audio_returns_default(self) -> None:
        a = RhythmAnalyzer()
        ctx = AnalysisContext(
            video_path=Path("/test.mp4"), audio_path=None,
            work_dir=Path("/tmp"), metadata=MOCK_META,
        )
        shot = Shot(shot_id=0, start_sec=0.0, end_sec=3.0, start_frame=0, end_frame=90, duration_sec=3.0)
        result = a.analyze_shot(shot, ctx)
        assert isinstance(result, RhythmResult)
        assert result.bpm == 0.0

    def test_with_preloaded_data(self) -> None:
        a = RhythmAnalyzer()
        # Simulate pre-loaded audio analysis results
        a._audio = np.zeros(16000 * 10)
        a._sr = 16000
        a._bpm = 120.0
        a._beats = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        a._onsets = np.array([0.4, 1.1, 2.0, 3.1])

        ctx = AnalysisContext(
            video_path=Path("/test.mp4"), audio_path=Path("/test.wav"),
            work_dir=Path("/tmp"), metadata=MOCK_META,
        )

        shot = Shot(shot_id=0, start_sec=1.0, end_sec=3.0, start_frame=30, end_frame=90, duration_sec=2.0)
        result = a.analyze_shot(shot, ctx)

        assert result.bpm == 120.0
        assert len(result.beats_in_shot) > 0
        assert all(1.0 <= b <= 3.0 for b in result.beats_in_shot)
        assert result.music_cue == "beat"

    def test_beat_aligned_detection(self) -> None:
        a = RhythmAnalyzer()
        a._audio = np.zeros(16000 * 10)
        a._sr = 16000
        a._bpm = 120.0
        a._beats = np.array([0.0, 0.5, 1.0, 3.0])  # Beat at 3.0 = shot end
        a._onsets = np.array([])

        ctx = AnalysisContext(
            video_path=Path("/test.mp4"), audio_path=Path("/test.wav"),
            work_dir=Path("/tmp"), metadata=MOCK_META,
        )

        shot = Shot(shot_id=0, start_sec=0.0, end_sec=3.0, start_frame=0, end_frame=90, duration_sec=3.0)
        result = a.analyze_shot(shot, ctx)
        assert result.beat_aligned is True

    def test_cleanup(self) -> None:
        a = RhythmAnalyzer()
        a._audio = np.zeros(100)
        a._beats = np.array([1.0])
        a.cleanup()
        assert a._audio is None
        assert a._beats is None
        assert a._bpm == 0.0
