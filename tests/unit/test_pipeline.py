"""Tests for reels.pipeline module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reels.pipeline import PipelineState, STAGES


class TestPipelineState:
    def test_init_creates_empty(self, tmp_path: Path) -> None:
        state = PipelineState(tmp_path)
        assert state.completed_stages == []

    def test_init_state(self, tmp_path: Path) -> None:
        state = PipelineState(tmp_path)
        state.init("test.mp4")
        assert state.completed_stages == []
        assert state.path.exists()

    def test_mark_complete(self, tmp_path: Path) -> None:
        state = PipelineState(tmp_path)
        state.init("test.mp4")
        state.mark_complete("ingest", {"video": "out.mp4"})
        assert "ingest" in state.completed_stages
        assert state.get_stage_data("ingest") == {"video": "out.mp4"}

    def test_should_skip(self, tmp_path: Path) -> None:
        state = PipelineState(tmp_path)
        state.init("test.mp4")
        state.mark_complete("ingest")
        assert state.should_skip("ingest") is True
        assert state.should_skip("segmentation") is False

    def test_reset(self, tmp_path: Path) -> None:
        state = PipelineState(tmp_path)
        state.init("test.mp4")
        state.mark_complete("ingest")
        state.reset()
        assert state.completed_stages == []
        assert not state.path.exists()

    def test_persistence(self, tmp_path: Path) -> None:
        state1 = PipelineState(tmp_path)
        state1.init("test.mp4")
        state1.mark_complete("ingest", {"key": "value"})

        # Re-load from disk
        state2 = PipelineState(tmp_path)
        assert "ingest" in state2.completed_stages
        assert state2.get_stage_data("ingest") == {"key": "value"}

    def test_mark_started(self, tmp_path: Path) -> None:
        state = PipelineState(tmp_path)
        state.init("test.mp4")
        state.mark_started("segmentation")
        data = json.loads(state.path.read_text())
        assert data["current_stage"] == "segmentation"

    def test_corrupt_file_handled(self, tmp_path: Path) -> None:
        state_path = tmp_path / "pipeline_state.json"
        state_path.write_text("{invalid json")
        state = PipelineState(tmp_path)
        assert state.completed_stages == []
