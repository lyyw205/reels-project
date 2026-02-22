"""Tests for reels.analysis.camera module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from reels.analysis.base import AnalysisContext
from reels.analysis.camera import CameraAnalyzer
from reels.models.analysis import CameraResult
from reels.models.metadata import VideoMetadata
from reels.models.shot import Shot


MOCK_META = VideoMetadata(
    source="test.mp4", duration_sec=10.0, fps=30.0,
    width=1080, height=1920, resolution="1080x1920",
)

MOCK_SHOT = Shot(
    shot_id=0, start_sec=0.0, end_sec=3.0,
    start_frame=0, end_frame=90, duration_sec=3.0,
)


class TestCameraAnalyzer:
    def test_init_defaults(self) -> None:
        a = CameraAnalyzer()
        assert a.name == "camera"
        assert a.sample_interval == 2
        assert a.motion_threshold == 2.0

    def test_init_from_config(self) -> None:
        cfg = {"analysis": {"camera": {"sample_interval_frames": 5, "motion_threshold": 3.0}}}
        a = CameraAnalyzer(cfg)
        assert a.sample_interval == 5
        assert a.motion_threshold == 3.0

    def test_classify_static(self) -> None:
        a = CameraAnalyzer()
        flow = np.array([0.1, 0.1])
        result = a._classify_motion(flow, avg_magnitude=0.5, shake_score=0.0)
        assert result == "static"

    def test_classify_pan_right(self) -> None:
        a = CameraAnalyzer()
        flow = np.array([5.0, 0.5])
        result = a._classify_motion(flow, avg_magnitude=5.0, shake_score=0.1)
        assert result == "pan_right"

    def test_classify_pan_left(self) -> None:
        a = CameraAnalyzer()
        flow = np.array([-5.0, 0.5])
        result = a._classify_motion(flow, avg_magnitude=5.0, shake_score=0.1)
        assert result == "pan_left"

    def test_classify_tilt_down(self) -> None:
        a = CameraAnalyzer()
        flow = np.array([0.3, 5.0])
        result = a._classify_motion(flow, avg_magnitude=5.0, shake_score=0.1)
        assert result == "tilt_down"

    def test_classify_handheld(self) -> None:
        a = CameraAnalyzer()
        flow = np.array([2.0, 2.0])
        result = a._classify_motion(flow, avg_magnitude=3.0, shake_score=0.6)
        assert result == "handheld"

    def test_shake_score_low_variance(self) -> None:
        a = CameraAnalyzer()
        vectors = [np.array([1.0, 0.0])] * 10
        score = a._compute_shake_score(vectors)
        assert score == 0.0

    def test_shake_score_high_variance(self) -> None:
        a = CameraAnalyzer()
        vectors = [np.array([float(i), float(-i)]) for i in range(-5, 5)]
        score = a._compute_shake_score(vectors)
        assert score > 0.0

    def test_analyze_shot_video_not_found(self) -> None:
        a = CameraAnalyzer()
        ctx = AnalysisContext(
            video_path=Path("/nonexistent.mp4"), audio_path=None,
            work_dir=Path("/tmp"), metadata=MOCK_META,
        )
        result = a.analyze_shot(MOCK_SHOT, ctx)
        assert isinstance(result, CameraResult)
        assert result.camera_type == "static"

    def test_classify_push_in(self) -> None:
        a = CameraAnalyzer()
        flow = np.array([1.0, 1.0])
        result = a._classify_motion(flow, avg_magnitude=3.0, shake_score=0.05, avg_divergence=1.5)
        assert result == "push_in"

    def test_classify_pull_out(self) -> None:
        a = CameraAnalyzer()
        flow = np.array([1.0, 1.0])
        result = a._classify_motion(flow, avg_magnitude=3.0, shake_score=0.05, avg_divergence=-1.5)
        assert result == "pull_out"

    def test_classify_gimbal_smooth_low_divergence(self) -> None:
        a = CameraAnalyzer()
        flow = np.array([1.0, 1.0])
        result = a._classify_motion(flow, avg_magnitude=3.0, shake_score=0.05, avg_divergence=0.2)
        assert result == "gimbal_smooth"

    def test_radial_divergence_uniform_outward(self) -> None:
        """Flow vectors pointing outward from center → positive divergence."""
        h, w = 64, 64
        flow = np.zeros((h, w, 2), dtype=np.float32)
        cx, cy = w / 2.0, h / 2.0
        ys, xs = np.mgrid[0:h, 0:w]
        flow[..., 0] = (xs - cx).astype(np.float32) * 0.1
        flow[..., 1] = (ys - cy).astype(np.float32) * 0.1
        div = CameraAnalyzer._compute_radial_divergence(flow)
        assert div > 0

    def test_radial_divergence_uniform_inward(self) -> None:
        """Flow vectors pointing inward toward center → negative divergence."""
        h, w = 64, 64
        flow = np.zeros((h, w, 2), dtype=np.float32)
        cx, cy = w / 2.0, h / 2.0
        ys, xs = np.mgrid[0:h, 0:w]
        flow[..., 0] = -(xs - cx).astype(np.float32) * 0.1
        flow[..., 1] = -(ys - cy).astype(np.float32) * 0.1
        div = CameraAnalyzer._compute_radial_divergence(flow)
        assert div < 0

    def test_cleanup_noop(self) -> None:
        a = CameraAnalyzer()
        a.cleanup()  # Should not raise
