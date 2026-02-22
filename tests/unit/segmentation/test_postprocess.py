"""Tests for reels.segmentation.postprocess module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from reels.models.metadata import VideoMetadata
from reels.models.shot import Shot, ShotBoundary
from reels.segmentation.postprocess import (
    boundaries_to_shots,
    merge_short_shots,
    merge_similar_shots,
)


MOCK_META = VideoMetadata(
    source="test.mp4",
    duration_sec=15.0,
    fps=30.0,
    width=1080,
    height=1920,
    resolution="1080x1920",
)


class TestBoundariesToShots:
    def test_no_boundaries_single_shot(self) -> None:
        shots = boundaries_to_shots([], MOCK_META)
        assert len(shots) == 1
        assert shots[0].start_sec == 0.0
        assert shots[0].end_sec == 15.0

    def test_single_boundary_at_zero(self) -> None:
        boundaries = [ShotBoundary(frame_number=0, timecode_sec=0.0)]
        shots = boundaries_to_shots(boundaries, MOCK_META)
        assert len(shots) == 1
        assert shots[0].duration_sec == 15.0

    def test_multiple_boundaries(self) -> None:
        boundaries = [
            ShotBoundary(frame_number=0, timecode_sec=0.0),
            ShotBoundary(frame_number=90, timecode_sec=3.0),
            ShotBoundary(frame_number=180, timecode_sec=6.0),
        ]
        shots = boundaries_to_shots(boundaries, MOCK_META)
        assert len(shots) == 3
        assert shots[0].start_sec == 0.0
        assert shots[0].end_sec == 3.0
        assert shots[1].start_sec == 3.0
        assert shots[1].end_sec == 6.0
        assert shots[2].start_sec == 6.0
        assert shots[2].end_sec == 15.0

    def test_shot_ids_sequential(self) -> None:
        boundaries = [
            ShotBoundary(frame_number=0, timecode_sec=0.0),
            ShotBoundary(frame_number=150, timecode_sec=5.0),
        ]
        shots = boundaries_to_shots(boundaries, MOCK_META)
        assert [s.shot_id for s in shots] == [0, 1]

    def test_unordered_boundaries_sorted(self) -> None:
        boundaries = [
            ShotBoundary(frame_number=180, timecode_sec=6.0),
            ShotBoundary(frame_number=0, timecode_sec=0.0),
            ShotBoundary(frame_number=90, timecode_sec=3.0),
        ]
        shots = boundaries_to_shots(boundaries, MOCK_META)
        assert shots[0].start_frame == 0
        assert shots[1].start_frame == 90
        assert shots[2].start_frame == 180


class TestMergeShortShots:
    def test_no_short_shots(self) -> None:
        shots = [
            Shot(shot_id=0, start_sec=0.0, end_sec=3.0, start_frame=0, end_frame=90, duration_sec=3.0),
            Shot(shot_id=1, start_sec=3.0, end_sec=6.0, start_frame=90, end_frame=180, duration_sec=3.0),
        ]
        result = merge_short_shots(shots, min_duration_sec=0.25)
        assert len(result) == 2

    def test_short_shot_merged_into_previous(self) -> None:
        shots = [
            Shot(shot_id=0, start_sec=0.0, end_sec=3.0, start_frame=0, end_frame=90, duration_sec=3.0),
            Shot(shot_id=1, start_sec=3.0, end_sec=3.1, start_frame=90, end_frame=93, duration_sec=0.1),
            Shot(shot_id=2, start_sec=3.1, end_sec=6.0, start_frame=93, end_frame=180, duration_sec=2.9),
        ]
        result = merge_short_shots(shots, min_duration_sec=0.25)
        assert len(result) == 2
        assert result[0].end_sec == 3.1
        assert result[0].duration_sec == pytest.approx(3.1, abs=0.01)

    def test_first_short_shot_kept_if_no_previous(self) -> None:
        shots = [
            Shot(shot_id=0, start_sec=0.0, end_sec=0.1, start_frame=0, end_frame=3, duration_sec=0.1),
            Shot(shot_id=1, start_sec=0.1, end_sec=5.0, start_frame=3, end_frame=150, duration_sec=4.9),
        ]
        result = merge_short_shots(shots, min_duration_sec=0.25)
        # First short shot has no previous, so it stays (2 total)
        assert len(result) == 2

    def test_ids_reassigned(self) -> None:
        shots = [
            Shot(shot_id=0, start_sec=0.0, end_sec=3.0, start_frame=0, end_frame=90, duration_sec=3.0),
            Shot(shot_id=1, start_sec=3.0, end_sec=3.1, start_frame=90, end_frame=93, duration_sec=0.1),
            Shot(shot_id=2, start_sec=3.1, end_sec=6.0, start_frame=93, end_frame=180, duration_sec=2.9),
        ]
        result = merge_short_shots(shots, min_duration_sec=0.25)
        assert [s.shot_id for s in result] == [0, 1]

    def test_empty_list(self) -> None:
        assert merge_short_shots([], min_duration_sec=0.25) == []


class TestMergeSimilarShots:
    def test_single_shot_unchanged(self) -> None:
        shots = [Shot(shot_id=0, start_sec=0.0, end_sec=5.0, start_frame=0, end_frame=150, duration_sec=5.0)]
        result = merge_similar_shots(shots, Path("test.mp4"), similarity_threshold=0.92)
        assert len(result) == 1

    def test_similar_shots_merged(self) -> None:
        shots = [
            Shot(shot_id=0, start_sec=0.0, end_sec=3.0, start_frame=0, end_frame=90, duration_sec=3.0),
            Shot(shot_id=1, start_sec=3.0, end_sec=6.0, start_frame=90, end_frame=180, duration_sec=3.0),
        ]

        # Mock read_frame_at to return identical frames → high SSIM
        white_frame = np.ones((100, 100, 3), dtype=np.uint8) * 255

        with patch("reels.segmentation.postprocess.read_frame_at", return_value=white_frame):
            result = merge_similar_shots(shots, Path("test.mp4"), similarity_threshold=0.92)

        assert len(result) == 1
        assert result[0].end_sec == 6.0

    def test_different_shots_not_merged(self) -> None:
        shots = [
            Shot(shot_id=0, start_sec=0.0, end_sec=3.0, start_frame=0, end_frame=90, duration_sec=3.0),
            Shot(shot_id=1, start_sec=3.0, end_sec=6.0, start_frame=90, end_frame=180, duration_sec=3.0),
        ]

        def mock_read(path: Path, frame_num: int) -> np.ndarray:
            if frame_num < 90:
                return np.zeros((100, 100, 3), dtype=np.uint8)  # black
            return np.ones((100, 100, 3), dtype=np.uint8) * 255  # white

        with patch("reels.segmentation.postprocess.read_frame_at", side_effect=mock_read):
            result = merge_similar_shots(shots, Path("test.mp4"), similarity_threshold=0.92)

        assert len(result) == 2
