"""Shot-related Pydantic models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ShotBoundary(BaseModel):
    """Shot boundary information."""

    frame_number: int
    timecode_sec: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class Shot(BaseModel):
    """Individual shot information."""

    shot_id: int
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    duration_sec: float = Field(ge=0.0)
    keyframe_paths: list[str] = Field(default_factory=list)


class SegmentationResult(BaseModel):
    """Result of shot segmentation."""

    shots: list[Shot]
    keyframe_dir: str
    total_shots: int = 0

    def model_post_init(self, __context: object) -> None:
        if self.total_shots == 0:
            self.total_shots = len(self.shots)
