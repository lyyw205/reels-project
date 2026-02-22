"""Tests for reels.analysis.place module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reels.analysis.base import AnalysisContext
from reels.analysis.place import PlaceAnalyzer
from reels.models.analysis import PlaceResult
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


class TestPlaceAnalyzer:
    def test_init_defaults(self) -> None:
        a = PlaceAnalyzer()
        assert a.name == "place"
        assert a.auto_discover is True
        assert a.batch_size == 16

    def test_init_from_config(self) -> None:
        cfg = {"analysis": {"place": {"batch_size": 8, "auto_discover": False}}}
        a = PlaceAnalyzer(cfg)
        assert a.batch_size == 8
        assert a.auto_discover is False

    def test_taxonomy_store_lazy_loaded(self, tmp_path: Path) -> None:
        cfg = {"analysis": {"place": {"taxonomy_path": str(tmp_path / "tax.json")}}}
        a = PlaceAnalyzer(cfg)
        assert a._taxonomy_store is None
        _ = a.taxonomy_store
        assert a._taxonomy_store is not None
        assert a.taxonomy_store.label_count > 0

    def test_no_keyframes_returns_default(self) -> None:
        a = PlaceAnalyzer()
        ctx = AnalysisContext(
            video_path=Path("/nonexistent.mp4"), audio_path=None,
            work_dir=Path("/tmp/nonexistent"), metadata=MOCK_META,
        )
        shot = MOCK_SHOT.model_copy()
        shot.keyframe_paths = []
        result = a.analyze_shot(shot, ctx)
        assert isinstance(result, PlaceResult)
        assert result.place_label == "other"

    def test_cleanup_flushes_taxonomy(self, tmp_path: Path) -> None:
        cfg = {"analysis": {"place": {"taxonomy_path": str(tmp_path / "tax.json")}}}
        a = PlaceAnalyzer(cfg)
        a._model = "mock"
        a._processor = "mock"
        # Access taxonomy store to initialize it
        store = a.taxonomy_store
        store.increment_count("bedroom")
        assert store._dirty is True

        a.cleanup()
        assert a._model is None
        assert a._processor is None
        # Taxonomy should have been flushed
        assert store._dirty is False

    def test_expand_candidates_from_filename(self, tmp_path: Path) -> None:
        cfg = {"analysis": {"place": {"taxonomy_path": str(tmp_path / "tax.json")}}}
        a = PlaceAnalyzer(cfg)
        candidates = a._expand_candidates("야외노천탕.mp4")
        assert "야외노천탕" in candidates
        assert len(candidates["야외노천탕"]) >= 2

    def test_expand_candidates_existing_label(self, tmp_path: Path) -> None:
        cfg = {"analysis": {"place": {"taxonomy_path": str(tmp_path / "tax.json")}}}
        a = PlaceAnalyzer(cfg)
        candidates = a._expand_candidates("bedroom.mp4")
        assert candidates == {}  # Already exists in taxonomy
