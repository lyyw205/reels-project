"""Tests for reels.config module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from reels.config import (
    CameraAnalysisSettings,
    DbSettings,
    IngestNormalizeSettings,
    PipelineSettings,
    PlaceAnalysisSettings,
    RhythmAnalysisSettings,
    SegmentationSettings,
    SpeechAnalysisSettings,
    SubtitleAnalysisSettings,
    SynthesisSettings,
    get_config,
    load_yaml_config,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestLoadYamlConfig:
    def test_load_default_config(self) -> None:
        cfg = load_yaml_config()
        assert isinstance(cfg, dict)
        assert "pipeline" in cfg

    def test_load_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        cfg = load_yaml_config(tmp_path / "nope.yaml")
        assert cfg == {}

    def test_load_custom_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "custom.yaml"
        p.write_text("pipeline:\n  log_level: DEBUG\n")
        cfg = load_yaml_config(p)
        assert cfg["pipeline"]["log_level"] == "DEBUG"


class TestGetConfig:
    def test_returns_dict(self) -> None:
        cfg = get_config()
        assert isinstance(cfg, dict)


class TestPipelineSettings:
    def test_defaults(self) -> None:
        s = PipelineSettings()
        assert s.work_dir == "./work"
        assert s.cache_enabled is True
        assert s.log_level == "INFO"
        assert s.max_video_duration_sec == 300
        assert s.cleanup_keyframes_after_synthesis is False

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REELS_PIPELINE_LOG_LEVEL", "DEBUG")
        s = PipelineSettings()
        assert s.log_level == "DEBUG"


class TestIngestNormalizeSettings:
    def test_defaults(self) -> None:
        s = IngestNormalizeSettings()
        assert s.strategy == "preserve_aspect"
        assert s.max_short_side == 1080
        assert s.target_fps == 30
        assert s.audio_sample_rate == 16000

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REELS_INGEST_TARGET_FPS", "60")
        s = IngestNormalizeSettings()
        assert s.target_fps == 60


class TestSegmentationSettings:
    def test_defaults(self) -> None:
        s = SegmentationSettings()
        assert s.detector == "pyscenedetect"
        assert s.threshold == 27.0
        assert s.min_shot_duration_sec == 0.25
        assert s.merge_similar_threshold == 0.92


class TestPlaceAnalysisSettings:
    def test_defaults(self) -> None:
        s = PlaceAnalysisSettings()
        assert "exterior" in s.taxonomy
        assert s.batch_size == 16


class TestCameraAnalysisSettings:
    def test_defaults(self) -> None:
        s = CameraAnalysisSettings()
        assert s.optical_flow_method == "farneback"
        assert s.sample_interval_frames == 2


class TestSubtitleAnalysisSettings:
    def test_defaults(self) -> None:
        s = SubtitleAnalysisSettings()
        assert s.engine == "easyocr"
        assert "ko" in s.languages
        assert s.exclude_bottom_ratio == 0.85


class TestSpeechAnalysisSettings:
    def test_defaults(self) -> None:
        s = SpeechAnalysisSettings()
        assert s.model == "small"
        assert s.compute_type == "int8"
        assert s.word_timestamps is True


class TestRhythmAnalysisSettings:
    def test_defaults(self) -> None:
        s = RhythmAnalysisSettings()
        assert s.hop_length == 512


class TestSynthesisSettings:
    def test_defaults(self) -> None:
        s = SynthesisSettings()
        assert s.template_version == "1.0"
        assert s.keyframe_format == "jpeg"


class TestDbSettings:
    def test_defaults(self) -> None:
        s = DbSettings()
        assert s.path == "./data/templates.db"
