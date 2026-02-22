"""Tests for reels.analysis.runner module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reels.analysis.base import AnalysisContext
from reels.analysis.runner import AnalysisRunner
from reels.models.analysis import CameraResult, PlaceResult
from reels.models.metadata import VideoMetadata
from reels.models.shot import Shot


MOCK_META = VideoMetadata(
    source="test.mp4", duration_sec=10.0, fps=30.0,
    width=1080, height=1920, resolution="1080x1920",
)

MOCK_SHOTS = [
    Shot(shot_id=0, start_sec=0.0, end_sec=3.0, start_frame=0, end_frame=90, duration_sec=3.0),
    Shot(shot_id=1, start_sec=3.0, end_sec=6.0, start_frame=90, end_frame=180, duration_sec=3.0),
]


def _make_context() -> AnalysisContext:
    return AnalysisContext(
        video_path=Path("/test.mp4"), audio_path=None,
        work_dir=Path("/tmp"), metadata=MOCK_META,
    )


class TestAnalysisRunner:
    def test_register_and_run(self) -> None:
        runner = AnalysisRunner()

        mock_analyzer = MagicMock()
        mock_analyzer.name = "test_analyzer"
        mock_analyzer.analyze_batch.return_value = [
            PlaceResult(place_label="bedroom", confidence=0.9),
            PlaceResult(place_label="bathroom", confidence=0.8),
        ]

        runner.register(mock_analyzer)
        results = runner.run_all(MOCK_SHOTS, _make_context())

        assert "test_analyzer" in results
        assert len(results["test_analyzer"]) == 2
        mock_analyzer.cleanup.assert_called_once()

    def test_multiple_analyzers(self) -> None:
        runner = AnalysisRunner()

        a1 = MagicMock()
        a1.name = "place"
        a1.analyze_batch.return_value = [PlaceResult(), PlaceResult()]

        a2 = MagicMock()
        a2.name = "camera"
        a2.analyze_batch.return_value = [CameraResult(), CameraResult()]

        runner.register(a1)
        runner.register(a2)
        results = runner.run_all(MOCK_SHOTS, _make_context())

        assert "place" in results
        assert "camera" in results
        a1.cleanup.assert_called_once()
        a2.cleanup.assert_called_once()

    def test_analyzer_failure_isolated(self) -> None:
        runner = AnalysisRunner()

        good = MagicMock()
        good.name = "good"
        good.analyze_batch.return_value = [PlaceResult()]

        bad = MagicMock()
        bad.name = "bad"
        bad.analyze_batch.side_effect = RuntimeError("model crash")

        runner.register(bad)
        runner.register(good)
        results = runner.run_all([MOCK_SHOTS[0]], _make_context())

        # Bad analyzer returns empty, good analyzer still runs
        assert results["bad"] == []
        assert len(results["good"]) == 1
        bad.cleanup.assert_called_once()
        good.cleanup.assert_called_once()

    def test_cleanup_failure_doesnt_block(self) -> None:
        runner = AnalysisRunner()

        analyzer = MagicMock()
        analyzer.name = "noisy"
        analyzer.analyze_batch.return_value = [PlaceResult()]
        analyzer.cleanup.side_effect = RuntimeError("cleanup boom")

        runner.register(analyzer)
        # Should not raise
        results = runner.run_all([MOCK_SHOTS[0]], _make_context())
        assert "noisy" in results
