"""Configuration: pydantic-settings with YAML defaults + env override."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import Field
from pydantic_settings import BaseSettings

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"


class PipelineSettings(BaseSettings):
    """Pipeline-level settings."""

    work_dir: str = "./work"
    cache_enabled: bool = True
    log_level: str = "INFO"
    max_video_duration_sec: int = 300
    cleanup_keyframes_after_synthesis: bool = False

    model_config = {"env_prefix": "REELS_PIPELINE_"}


class IngestNormalizeSettings(BaseSettings):
    """Ingest normalization settings."""

    strategy: str = "preserve_aspect"
    max_short_side: int = 1080
    target_fps: int = 30
    audio_sample_rate: int = 16000
    video_codec: str = "libx264"
    audio_codec: str = "aac"

    model_config = {"env_prefix": "REELS_INGEST_"}


class SegmentationSettings(BaseSettings):
    """Shot segmentation settings."""

    detector: str = "pyscenedetect"
    threshold: float = 27.0
    min_shot_duration_sec: float = 0.25
    merge_similar_threshold: float = 0.92

    model_config = {"env_prefix": "REELS_SEGMENTATION_"}


class PlaceAnalysisSettings(BaseSettings):
    """Place/subject analysis settings."""

    model: str = "openai/clip-vit-base-patch32"
    keyframes_per_shot: int = 3
    taxonomy: list[str] = Field(default_factory=lambda: [
        "exterior", "lobby", "bedroom", "bathroom",
        "pool", "restaurant", "view", "amenity", "other",
    ])
    batch_size: int = 16

    model_config = {"env_prefix": "REELS_ANALYSIS_PLACE_"}


class CameraAnalysisSettings(BaseSettings):
    """Camera motion analysis settings."""

    optical_flow_method: str = "farneback"
    sample_interval_frames: int = 2
    motion_threshold: float = 2.0

    model_config = {"env_prefix": "REELS_ANALYSIS_CAMERA_"}


class SubtitleAnalysisSettings(BaseSettings):
    """Subtitle/OCR analysis settings."""

    engine: str = "easyocr"
    languages: list[str] = Field(default_factory=lambda: ["ko", "en"])
    confidence_threshold: float = 0.6
    dedup_iou_threshold: float = 0.5
    voting_window_frames: int = 5
    exclude_bottom_ratio: float = 0.85

    model_config = {"env_prefix": "REELS_ANALYSIS_SUBTITLE_"}


class SpeechAnalysisSettings(BaseSettings):
    """Speech recognition settings."""

    model: str = "small"
    language: str = "ko"
    compute_type: str = "int8"
    word_timestamps: bool = True
    vad_filter: bool = True

    model_config = {"env_prefix": "REELS_ANALYSIS_SPEECH_"}


class RhythmAnalysisSettings(BaseSettings):
    """Rhythm/beat analysis settings."""

    hop_length: int = 512
    onset_strength_threshold: float = 0.5
    beat_tracking_method: str = "librosa"

    model_config = {"env_prefix": "REELS_ANALYSIS_RHYTHM_"}


class SynthesisSettings(BaseSettings):
    """Template synthesis settings."""

    template_version: str = "1.0"
    include_keyframe_images: bool = True
    overlay_position_normalize: bool = True
    keyframe_format: str = "jpeg"
    keyframe_quality: int = 85

    model_config = {"env_prefix": "REELS_SYNTHESIS_"}


class DbSettings(BaseSettings):
    """Database settings."""

    path: str = "./data/templates.db"

    model_config = {"env_prefix": "REELS_DB_"}


def load_yaml_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load YAML config file."""
    path = config_path or _DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load config from YAML. Environment variables override via pydantic-settings."""
    return load_yaml_config(config_path)


# Alias for backward compatibility
load_config = get_config
