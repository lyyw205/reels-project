"""Claude Vision API backend for accommodation image analysis."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = """이 숙소 이미지를 분석해주세요.

1. 장면 분류: (exterior/lobby/bedroom/bathroom/terrace/pool/restaurant/view/amenity/other)
2. 발견된 특색/어메니티 (있는 것만, JSON 배열):

각 항목:
- tag: 한국어 태그 (예: "노천탕", "오션뷰")
- tag_en: 영어 태그 (예: "outdoor_bath", "ocean_view")
- category: scene/amenity/view/dining/activity
- confidence: 0.0-1.0 (확실할수록 높게)
- description: 판단 근거 한 문장

확실하지 않은 것은 신뢰도를 낮게. 없는 것을 있다고 하지 마세요.
JSON 배열만 출력해주세요."""


class ClaudeVisionBackend:
    """Claude Vision API backend."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("production", {}).get("feature_extraction", {})
        self.model = cfg.get("model", "claude-sonnet-4-20250514")
        self.max_features_per_image = cfg.get("max_features_per_image", 5)
        self._client = None  # Lazy init

    def _get_client(self):
        """Lazy-init Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic()
            except ImportError:
                raise RuntimeError(
                    "anthropic package required. Install: pip install anthropic"
                )
        return self._client

    async def analyze_image(
        self, image: Path, prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Analyze single image, return list of feature dicts."""
        client = self._get_client()
        prompt = prompt or _DEFAULT_PROMPT

        # Read and encode image
        image_data = image.read_bytes()
        media_type = self._guess_media_type(image)
        b64 = base64.standard_b64encode(image_data).decode("utf-8")

        try:
            response = await client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return self._parse_response(response)
        except Exception as e:
            logger.error("Claude Vision API error for %s: %s", image.name, e)
            return []

    async def analyze_batch(
        self, images: list[Path], prompt: str | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Analyze multiple images (called sequentially by FeatureExtractor's semaphore)."""
        results = []
        for img in images:
            result = await self.analyze_image(img, prompt)
            results.append(result)
        return results

    def cleanup(self) -> None:
        """Release client resources."""
        self._client = None

    def _parse_response(self, response) -> list[dict[str, Any]]:
        """Extract JSON array from Claude response text."""
        text = response.content[0].text.strip()
        # Try to find JSON array in response
        # Handle cases where Claude wraps JSON in markdown code blocks
        if "```" in text:
            # Extract content between code fences
            parts = text.split("```")
            for part in parts[1::2]:  # odd indices are inside fences
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    parsed = json.loads(part)
                    if isinstance(parsed, list):
                        return parsed[:self.max_features_per_image]
                except json.JSONDecodeError:
                    continue

        # Try direct parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed[:self.max_features_per_image]
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            logger.warning("Failed to parse VLM response as JSON: %s...", text[:100])
            return []

        return []

    @staticmethod
    def _guess_media_type(path: Path) -> str:
        suffix = path.suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")
