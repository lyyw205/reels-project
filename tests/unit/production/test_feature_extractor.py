"""Tests for reels.production.feature_extractor module."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reels.production.feature_extractor import FeatureExtractor
from reels.production.models import Feature, FeatureCategory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature(
    tag: str = "노천탕",
    tag_en: str = "outdoor_bath",
    confidence: float = 0.80,
    evidence_images: list[str] | None = None,
    category: FeatureCategory = FeatureCategory.AMENITY,
    description: str = "",
) -> Feature:
    return Feature(
        tag=tag,
        tag_en=tag_en,
        confidence=confidence,
        evidence_images=evidence_images or [],
        category=category,
        description=description,
    )


def _make_extractor() -> FeatureExtractor:
    """Return a FeatureExtractor with ClaudeVisionBackend and ResponseCache mocked out."""
    with (
        patch("reels.production.feature_extractor.ClaudeVisionBackend"),
        patch("reels.production.feature_extractor.ResponseCache"),
    ):
        extractor = FeatureExtractor()
    return extractor


# ---------------------------------------------------------------------------
# _merge_features
# ---------------------------------------------------------------------------

class TestMergeFeatures:
    def test_merge_same_tag_combines_evidence(self) -> None:
        """Two features with same tag_en from different images → merged, evidence combined."""
        extractor = _make_extractor()
        f1 = _make_feature(tag_en="outdoor_bath", evidence_images=["img1.jpg"])
        f2 = _make_feature(tag_en="outdoor_bath", evidence_images=["img2.jpg"])

        result = extractor._merge_features([f1, f2])

        assert len(result) == 1
        assert set(result[0].evidence_images) == {"img1.jpg", "img2.jpg"}

    def test_merge_boosts_confidence(self) -> None:
        """3 images finding same feature → confidence boosted by 0.10 (2 extra * 0.05)."""
        extractor = _make_extractor()
        base_conf = 0.70
        features = [
            _make_feature(tag_en="ocean_view", confidence=base_conf, evidence_images=["a.jpg"]),
            _make_feature(tag_en="ocean_view", confidence=0.60, evidence_images=["b.jpg"]),
            _make_feature(tag_en="ocean_view", confidence=0.50, evidence_images=["c.jpg"]),
        ]

        result = extractor._merge_features(features)

        assert len(result) == 1
        # best confidence 0.70, 3 images → boost = (3-1)*0.05 = 0.10
        assert result[0].confidence == pytest.approx(0.80, abs=1e-3)

    def test_merge_different_tags_kept_separate(self) -> None:
        """Different tag_en values → not merged, both retained."""
        extractor = _make_extractor()
        f1 = _make_feature(tag_en="outdoor_bath", evidence_images=["img1.jpg"])
        f2 = _make_feature(tag_en="ocean_view", evidence_images=["img2.jpg"])

        result = extractor._merge_features([f1, f2])

        assert len(result) == 2
        tags = {f.tag_en for f in result}
        assert tags == {"outdoor_bath", "ocean_view"}


# ---------------------------------------------------------------------------
# _rank_features
# ---------------------------------------------------------------------------

class TestRankFeatures:
    def test_rank_by_weighted_score(self) -> None:
        """AMENITY(weight=1.0) with conf=0.6 scores 0.6; SCENE(weight=0.5) with conf=0.8 scores 0.4.
        AMENITY should rank higher despite lower raw confidence."""
        extractor = _make_extractor()
        amenity = _make_feature(tag_en="pool", confidence=0.6, category=FeatureCategory.AMENITY)
        scene = _make_feature(tag_en="lobby", confidence=0.8, category=FeatureCategory.SCENE)

        result = extractor._rank_features([scene, amenity])

        assert result[0].tag_en == "pool"
        assert result[1].tag_en == "lobby"


# ---------------------------------------------------------------------------
# _parse_raw_features
# ---------------------------------------------------------------------------

class TestParseRawFeatures:
    def test_parse_raw_features_valid(self) -> None:
        """Valid dict with all fields → Feature created correctly."""
        extractor = _make_extractor()
        raw = [{
            "tag": "노천탕",
            "tag_en": "outdoor_bath",
            "category": "amenity",
            "confidence": 0.85,
            "description": "실외 욕조 확인됨",
        }]

        result = extractor._parse_raw_features(raw, "test.jpg")

        assert len(result) == 1
        f = result[0]
        assert f.tag == "노천탕"
        assert f.tag_en == "outdoor_bath"
        assert f.category == FeatureCategory.AMENITY
        assert f.confidence == pytest.approx(0.85)
        assert f.evidence_images == ["test.jpg"]
        assert f.description == "실외 욕조 확인됨"

    def test_parse_raw_features_malformed(self) -> None:
        """Items that raise ValueError/TypeError are skipped with a warning."""
        extractor = _make_extractor()
        # confidence=None will cause float(None) → TypeError
        raw = [{"tag": "bad", "tag_en": "bad", "confidence": None}]

        import logging
        with patch.object(
            extractor, "_parse_raw_features", wraps=extractor._parse_raw_features
        ):
            # confidence=None triggers TypeError in float(None)
            result = extractor._parse_raw_features(raw, "test.jpg")

        # Malformed item skipped → empty result
        assert result == []

    def test_parse_raw_features_clamps_confidence(self) -> None:
        """confidence > 1.0 is clamped to 1.0."""
        extractor = _make_extractor()
        raw = [{
            "tag": "뷰",
            "tag_en": "mountain_view",
            "category": "view",
            "confidence": 1.5,
            "description": "",
        }]

        result = extractor._parse_raw_features(raw, "test.jpg")

        assert len(result) == 1
        assert result[0].confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# extract (async, integration of the full flow)
# ---------------------------------------------------------------------------

class TestExtract:
    def test_extract_tolerates_partial_failure(self) -> None:
        """Backend that fails on some images → successful results are preserved."""
        extractor = _make_extractor()

        good_features = [
            _make_feature(tag_en="outdoor_bath", evidence_images=["good.jpg"]),
        ]

        async def fake_analyze(image_path: Path) -> list[Feature]:
            if image_path.name == "bad.jpg":
                raise RuntimeError("API error")
            return good_features

        # Patch _analyze_with_limit directly to bypass cache/semaphore complexity
        extractor._analyze_with_limit = fake_analyze  # type: ignore[method-assign]

        images = [Path("good.jpg"), Path("bad.jpg")]
        result = asyncio.run(extractor.extract(images))

        # good.jpg contributed features; bad.jpg failure is tolerated
        assert len(result) >= 1
        assert any(f.tag_en == "outdoor_bath" for f in result)
