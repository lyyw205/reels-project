"""Feature extractor — analyzes accommodation images to discover key features."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from reels.production.backends.claude_vision import ClaudeVisionBackend
from reels.production.cache import ResponseCache
from reels.production.models import (
    AccommodationInput,
    Feature,
    FeatureCategory,
)

logger = logging.getLogger(__name__)

# Category importance for ranking
_CATEGORY_WEIGHTS = {
    FeatureCategory.AMENITY: 1.0,    # Highest value for unique amenities
    FeatureCategory.VIEW: 0.9,
    FeatureCategory.DINING: 0.8,
    FeatureCategory.ACTIVITY: 0.7,
    FeatureCategory.SCENE: 0.5,      # Basic scenes are least differentiating
}


class FeatureExtractor:
    """Extract accommodation features from images using VLM."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("production", {}).get("feature_extraction", {})
        self._backend = ClaudeVisionBackend(config)
        self._cache = ResponseCache(cfg.get("cache_dir", ".cache/vlm"))
        self._semaphore = asyncio.Semaphore(cfg.get("max_concurrent", 5))
        self._merge_threshold = cfg.get("merge_threshold", 0.8)

    async def extract(
        self, images: list[Path], context: AccommodationInput | None = None,
    ) -> list[Feature]:
        """Analyze N images in parallel and return merged, ranked features.

        Uses semaphore to limit concurrent API calls.
        Tolerates partial failures (logs warning, continues with successful results).
        """
        tasks = [self._analyze_with_limit(img) for img in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_features: list[Feature] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("Image %s analysis failed: %s", images[i].name, result)
                continue
            all_features.extend(result)

        if not all_features:
            logger.warning("No features extracted from %d images", len(images))
            return []

        merged = self._merge_features(all_features)
        ranked = self._rank_features(merged)
        logger.info("Extracted %d features from %d images", len(ranked), len(images))
        return ranked

    async def _analyze_with_limit(self, image_path: Path) -> list[Feature]:
        """Analyze single image with semaphore + cache + retry."""
        # Check cache first
        cache_key = self._cache.hash_file(image_path)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return [Feature(**f) for f in cached]

        # Rate-limited API call with retry
        async with self._semaphore:
            for attempt in range(3):
                try:
                    raw = await self._backend.analyze_image(image_path)
                    features = self._parse_raw_features(raw, image_path.name)
                    # Cache successful result
                    self._cache.put(cache_key, [f.model_dump() for f in features])
                    return features
                except Exception as e:
                    if attempt < 2:
                        wait = 2 ** attempt
                        logger.warning(
                            "Retry %d for %s (wait %ds): %s",
                            attempt + 1, image_path.name, wait, e,
                        )
                        await asyncio.sleep(wait)
                    else:
                        raise

        return []  # unreachable but satisfies type checker

    def _parse_raw_features(
        self, raw: list[dict[str, Any]], image_name: str,
    ) -> list[Feature]:
        """Convert raw VLM dicts to Feature models with validation."""
        features = []
        for item in raw:
            try:
                category = item.get("category", "scene")
                if category not in FeatureCategory.__members__.values():
                    category = "scene"

                features.append(Feature(
                    tag=item.get("tag", ""),
                    tag_en=item.get("tag_en", ""),
                    confidence=max(0.0, min(1.0, float(item.get("confidence", 0.5)))),
                    evidence_images=[image_name],
                    description=item.get("description", ""),
                    category=FeatureCategory(category),
                ))
            except (ValueError, TypeError) as e:
                logger.warning("Skipping malformed feature: %s — %s", item, e)
                continue
        return features

    def _merge_features(self, features: list[Feature]) -> list[Feature]:
        """Merge features with same tag_en across images.

        When the same feature appears in multiple images:
        - Confidence increases (multi-evidence)
        - Evidence images are combined
        - Best description is kept
        """
        groups: dict[str, list[Feature]] = defaultdict(list)
        for f in features:
            key = f.tag_en.lower().strip()
            if key:
                groups[key].append(f)

        merged = []
        for key, group in groups.items():
            if not group:
                continue

            # Take the one with highest confidence as base
            best = max(group, key=lambda f: f.confidence)

            # Combine evidence images (deduplicate)
            all_evidence = []
            seen = set()
            for f in group:
                for img in f.evidence_images:
                    if img not in seen:
                        all_evidence.append(img)
                        seen.add(img)

            # Boost confidence based on number of images that found this feature
            image_count = len(all_evidence)
            boosted_conf = min(1.0, best.confidence + (image_count - 1) * 0.05)

            merged.append(Feature(
                tag=best.tag,
                tag_en=best.tag_en,
                confidence=round(boosted_conf, 3),
                evidence_images=all_evidence,
                description=best.description,
                category=best.category,
            ))

        return merged

    def _rank_features(self, features: list[Feature]) -> list[Feature]:
        """Rank features by weighted score (confidence * category weight)."""
        def score(f: Feature) -> float:
            weight = _CATEGORY_WEIGHTS.get(f.category, 0.5)
            return f.confidence * weight

        return sorted(features, key=score, reverse=True)

    def cleanup(self) -> None:
        """Release backend resources."""
        self._backend.cleanup()
