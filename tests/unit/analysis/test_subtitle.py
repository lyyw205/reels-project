"""Tests for reels.analysis.subtitle module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from reels.analysis.base import AnalysisContext
from reels.analysis.subtitle import SubtitleAnalyzer
from reels.models.analysis import SubtitleResult
from reels.models.metadata import VideoMetadata
from reels.models.shot import Shot


MOCK_META = VideoMetadata(
    source="test.mp4", duration_sec=10.0, fps=30.0,
    width=1080, height=1920, resolution="1080x1920",
)


class TestSubtitleAnalyzer:
    def test_init_defaults(self) -> None:
        a = SubtitleAnalyzer()
        assert a.name == "subtitle"
        assert a.engine == "easyocr"
        assert "ko" in a.languages
        assert a.exclude_bottom_ratio == 0.85

    def test_init_from_config(self) -> None:
        cfg = {"analysis": {"subtitle": {"engine": "paddleocr", "languages": ["en"]}}}
        a = SubtitleAnalyzer(cfg)
        assert a.engine == "paddleocr"
        assert a.languages == ["en"]

    def test_majority_vote_basic(self) -> None:
        a = SubtitleAnalyzer()
        shot = Shot(shot_id=0, start_sec=0.0, end_sec=3.0, start_frame=0, end_frame=90, duration_sec=3.0)
        # Text appears in 3 out of 4 frames → passes majority vote
        detections = [
            [{"text": "Hello", "box": {"x": 0.1, "y": 0.7, "w": 0.8, "h": 0.1}, "confidence": 0.9}],
            [{"text": "Hello", "box": {"x": 0.1, "y": 0.7, "w": 0.8, "h": 0.1}, "confidence": 0.85}],
            [{"text": "Hello", "box": {"x": 0.1, "y": 0.7, "w": 0.8, "h": 0.1}, "confidence": 0.88}],
            [],
        ]
        entries = a._majority_vote(detections, shot)
        assert len(entries) == 1
        assert entries[0].text == "Hello"
        assert entries[0].confidence == 0.9  # Best confidence kept

    def test_majority_vote_filters_rare(self) -> None:
        a = SubtitleAnalyzer()
        shot = Shot(shot_id=0, start_sec=0.0, end_sec=3.0, start_frame=0, end_frame=90, duration_sec=3.0)
        # Text appears in 1 out of 4 frames → below threshold
        detections = [
            [{"text": "Noise", "box": {}, "confidence": 0.7}],
            [],
            [],
            [],
        ]
        entries = a._majority_vote(detections, shot)
        assert len(entries) == 0

    def test_majority_vote_empty(self) -> None:
        a = SubtitleAnalyzer()
        shot = Shot(shot_id=0, start_sec=0.0, end_sec=3.0, start_frame=0, end_frame=90, duration_sec=3.0)
        entries = a._majority_vote([], shot)
        assert entries == []

    def test_cleanup(self) -> None:
        a = SubtitleAnalyzer()
        a._reader = "mock"
        a.cleanup()
        assert a._reader is None

    def test_analyze_shot_no_video(self) -> None:
        a = SubtitleAnalyzer()
        ctx = AnalysisContext(
            video_path=Path("/nonexistent.mp4"), audio_path=None,
            work_dir=Path("/tmp"), metadata=MOCK_META,
        )
        shot = Shot(shot_id=0, start_sec=0.0, end_sec=3.0, start_frame=0, end_frame=90, duration_sec=3.0)
        # Mock the reader to avoid loading real OCR
        a._reader = MagicMock()
        result = a.analyze_shot(shot, ctx)
        assert isinstance(result, SubtitleResult)
