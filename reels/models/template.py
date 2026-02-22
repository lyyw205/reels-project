"""Template Pydantic models - final output schema with StrEnum types."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class OverlayKind(StrEnum):
    CAPTION = "caption"
    TITLE = "title"
    CTA = "cta"
    HASHTAG = "hashtag"
    WATERMARK = "watermark"


class CameraType(StrEnum):
    STATIC = "static"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    PUSH_IN = "push_in"
    PULL_OUT = "pull_out"
    HANDHELD = "handheld"
    GIMBAL_SMOOTH = "gimbal_smooth"


class TransitionType(StrEnum):
    CUT = "cut"
    CUT_ON_BEAT = "cut_on_beat"
    DISSOLVE = "dissolve"
    FADE = "fade"


class OverlayStyle(BaseModel):
    fill: str = "#FFFFFF"
    stroke: str | None = None
    shadow: bool = False
    font_size_ratio: float | None = None


class BoundingBox(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(ge=0.0, le=1.0)
    h: float = Field(ge=0.0, le=1.0)


class Overlay(BaseModel):
    kind: OverlayKind
    start_sec: float
    end_sec: float
    box: BoundingBox
    style: OverlayStyle = Field(default_factory=OverlayStyle)
    text: str
    animation: str | None = None


class CameraInfo(BaseModel):
    type: CameraType = CameraType.STATIC
    shake_score: float = Field(ge=0.0, le=1.0, default=0.0)
    speed_factor: float = 1.0


class AudioInfo(BaseModel):
    has_speech: bool = False
    speech_text: str | None = None
    vo_ref: str | None = None
    music_cue: str | None = None
    beat_aligned: bool = False


class EditInfo(BaseModel):
    speed: float = 1.0
    transition_out: TransitionType = TransitionType.CUT


class TemplateShot(BaseModel):
    shot_id: int
    start_sec: float
    end_sec: float
    duration_sec: float
    place_label: str = "other"
    camera: CameraInfo = Field(default_factory=CameraInfo)
    keyframe_paths: list[str] = Field(default_factory=list)
    overlays: list[Overlay] = Field(default_factory=list)
    audio: AudioInfo = Field(default_factory=AudioInfo)
    edit: EditInfo = Field(default_factory=EditInfo)
    bpm: float | None = None


class TemplateFormat(BaseModel):
    aspect: str = "9:16"
    fps: int = 30
    width: int = 1080
    height: int = 1920


class TemplateMetadata(BaseModel):
    source_platform: str | None = None
    original_resolution: str | None = None
    processing_time_sec: float | None = None
    analyzer_versions: dict[str, str] = Field(default_factory=dict)


class Template(BaseModel):
    template_id: str
    source_url: str | None = None
    format: TemplateFormat = Field(default_factory=TemplateFormat)
    total_duration_sec: float
    shot_count: int
    shots: list[TemplateShot]
    metadata: TemplateMetadata = Field(default_factory=TemplateMetadata)
