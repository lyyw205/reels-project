"""Video and audio metadata models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    """Metadata extracted from video via ffprobe."""

    source: str
    duration_sec: float = Field(ge=0.0)
    fps: float = Field(gt=0.0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    resolution: str
    bitrate_kbps: float | None = None
    has_audio: bool = True
    codec: str | None = None


class AudioMetadata(BaseModel):
    """Metadata for extracted audio track."""

    path: str
    sample_rate: int
    channels: int = 1
    duration_sec: float = Field(ge=0.0)
    codec: str | None = None


class NormalizeResult(BaseModel):
    """Result of video normalization."""

    video_path: str
    audio_path: str | None = None
    metadata: VideoMetadata


class IngestResult(BaseModel):
    """Result of full ingest pipeline."""

    original_path: str
    normalized_video: str
    audio_path: str | None = None
    metadata: VideoMetadata
