"""Pydantic data models for reels pipeline."""

from reels.models.metadata import AudioMetadata, VideoMetadata
from reels.models.shot import Shot, ShotBoundary
from reels.models.template import (
    AudioInfo,
    BoundingBox,
    CameraInfo,
    CameraType,
    EditInfo,
    Overlay,
    OverlayKind,
    OverlayStyle,
    Template,
    TemplateFormat,
    TemplateMetadata,
    TemplateShot,
    TransitionType,
)

__all__ = [
    "AudioInfo",
    "AudioMetadata",
    "BoundingBox",
    "CameraInfo",
    "CameraType",
    "EditInfo",
    "Overlay",
    "OverlayKind",
    "OverlayStyle",
    "Shot",
    "ShotBoundary",
    "Template",
    "TemplateFormat",
    "TemplateMetadata",
    "TemplateShot",
    "TransitionType",
    "VideoMetadata",
]
