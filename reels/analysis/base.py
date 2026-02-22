"""Analyzer Protocol and shared context."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from reels.models.metadata import VideoMetadata
from reels.models.shot import Shot

T = TypeVar("T", bound=BaseModel)


@dataclass
class AnalysisContext:
    """Shared context passed to all analyzers."""

    video_path: Path
    audio_path: Path | None
    work_dir: Path
    metadata: VideoMetadata
    config: dict[str, Any] = field(default_factory=dict)


class Analyzer(Protocol[T]):
    """Protocol that all analyzers must implement."""

    @property
    def name(self) -> str:
        """Analyzer name for logging and result keys."""
        ...

    def analyze_shot(self, shot: Shot, context: AnalysisContext) -> T:
        """Analyze a single shot. Returns a type-safe Pydantic result."""
        ...

    def analyze_batch(self, shots: list[Shot], context: AnalysisContext) -> list[T]:
        """Analyze multiple shots."""
        ...

    def cleanup(self) -> None:
        """Release resources (unload models, free memory)."""
        ...
