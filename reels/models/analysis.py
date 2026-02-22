"""Type-safe Pydantic result models for each analyzer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PlaceResult(BaseModel):
    """Result from PlaceAnalyzer (CLIP-based place classification)."""

    place_label: str = "other"
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    top_labels: list[tuple[str, float]] = Field(default_factory=list)


class CameraResult(BaseModel):
    """Result from CameraAnalyzer (optical flow-based motion classification)."""

    camera_type: str = "static"
    shake_score: float = Field(ge=0.0, le=1.0, default=0.0)
    avg_magnitude: float = 0.0
    dominant_direction: list[float] = Field(default_factory=list)


class SubtitleEntry(BaseModel):
    """Single detected subtitle/text overlay."""

    text: str
    box: dict[str, float] = Field(default_factory=dict)
    style: dict[str, Any] = Field(default_factory=dict)
    start_sec: float = 0.0
    end_sec: float = 0.0
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class SubtitleResult(BaseModel):
    """Result from SubtitleAnalyzer (OCR-based text extraction)."""

    texts: list[SubtitleEntry] = Field(default_factory=list)


class SpeechWord(BaseModel):
    """Word-level speech recognition result."""

    word: str
    start: float
    end: float
    probability: float = 0.0


class SpeechSegment(BaseModel):
    """Segment-level speech recognition result."""

    text: str
    start: float
    end: float
    words: list[SpeechWord] = Field(default_factory=list)


class SpeechResult(BaseModel):
    """Result from SpeechAnalyzer (whisper-based speech recognition)."""

    has_speech: bool = False
    segments: list[SpeechSegment] = Field(default_factory=list)


class RhythmResult(BaseModel):
    """Result from RhythmAnalyzer (librosa-based beat/onset analysis)."""

    bpm: float = 0.0
    beat_aligned: bool = False
    music_cue: str | None = None
    beats_in_shot: list[float] = Field(default_factory=list)
    onsets_in_shot: list[float] = Field(default_factory=list)
