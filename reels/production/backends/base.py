"""VLM/LLM backend Protocol interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class VLMBackend(Protocol):
    """Vision-Language Model backend interface."""

    async def analyze_image(self, image: Path, prompt: str) -> list[dict[str, Any]]: ...
    async def analyze_batch(self, images: list[Path], prompt: str) -> list[list[dict[str, Any]]]: ...
    def cleanup(self) -> None: ...


class LLMBackend(Protocol):
    """LLM backend interface for text generation."""

    async def generate(self, prompt: str, system: str | None = None) -> str: ...
