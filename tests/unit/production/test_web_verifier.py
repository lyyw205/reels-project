"""Tests for reels.production.web_verifier module."""

from __future__ import annotations

import asyncio

import pytest

from reels.production.models import (
    ClaimLevel,
    FeatureCategory,
    VerifiedFeature,
    WebEvidence,
)
from reels.production.web_verifier import WebVerifier


def _make_verified(
    tag: str = "노천탕",
    tag_en: str = "outdoor_bath",
    confidence: float = 0.60,
    claim_level: ClaimLevel = ClaimLevel.PROBABLE,
    web_evidence: list[WebEvidence] | None = None,
    copy_tone: str = "암시",
) -> VerifiedFeature:
    return VerifiedFeature(
        tag=tag,
        tag_en=tag_en,
        confidence=confidence,
        evidence_images=[],
        category=FeatureCategory.AMENITY,
        claim_level=claim_level,
        web_evidence=web_evidence or [],
        copy_tone=copy_tone,
    )


class TestVerifySkipConditions:
    def test_verify_skips_when_disabled(self) -> None:
        config = {"production": {"web_verification": {"enabled": False}}}
        verifier = WebVerifier(config=config)
        features = [_make_verified()]
        result = asyncio.run(verifier.verify(features, accommodation_name="호텔A"))
        assert result is features  # Same object returned unchanged

    def test_verify_skips_when_no_name(self) -> None:
        verifier = WebVerifier()
        features = [_make_verified()]
        result = asyncio.run(verifier.verify(features, accommodation_name=None))
        assert result is features


class TestShouldVerify:
    def test_should_verify_confirmed_false(self) -> None:
        verifier = WebVerifier()
        feature = _make_verified(claim_level=ClaimLevel.CONFIRMED, confidence=0.85)
        assert verifier._should_verify(feature, is_main=True) is False

    def test_should_verify_suggestive_false(self) -> None:
        verifier = WebVerifier()
        feature = _make_verified(claim_level=ClaimLevel.SUGGESTIVE, confidence=0.30)
        assert verifier._should_verify(feature, is_main=True) is False

    def test_should_verify_probable_main_true(self) -> None:
        verifier = WebVerifier()
        feature = _make_verified(claim_level=ClaimLevel.PROBABLE, confidence=0.60)
        assert verifier._should_verify(feature, is_main=True) is True

    def test_should_verify_probable_not_main_false(self) -> None:
        verifier = WebVerifier()
        feature = _make_verified(claim_level=ClaimLevel.PROBABLE, confidence=0.60)
        assert verifier._should_verify(feature, is_main=False) is False


class TestUpdateWithEvidence:
    def test_update_with_evidence_boosts_confidence(self) -> None:
        verifier = WebVerifier()
        feature = _make_verified(confidence=0.60)
        evidence = [WebEvidence(
            claim="호텔A has 노천탕",
            url="https://example.com",
            snippet="노천탕 at 호텔A",
            confidence=0.8,
        )]
        updated = verifier._update_with_evidence(feature, evidence)
        assert updated.confidence == pytest.approx(0.8)
        assert len(updated.web_evidence) == 1
        assert updated.web_evidence[0].confidence == pytest.approx(0.8)

    def test_update_with_empty_evidence(self) -> None:
        verifier = WebVerifier()
        feature = _make_verified(confidence=0.60)
        updated = verifier._update_with_evidence(feature, [])
        assert updated is feature  # Unchanged


class TestMaxQueriesLimit:
    def test_max_queries_limit(self) -> None:
        config = {"production": {"web_verification": {"max_queries": 1}}}
        verifier = WebVerifier(config=config)

        # 3 PROBABLE features, all "main" (indices 0 and 1 are main)
        features = [
            _make_verified(tag="노천탕"),
            _make_verified(tag="수영장"),
            _make_verified(tag="사우나"),
        ]

        evidence = [WebEvidence(
            claim="found",
            url="https://example.com",
            snippet="found",
            confidence=0.7,
        )]

        call_count = 0

        async def fake_search(name: str, tag: str) -> list[WebEvidence]:
            nonlocal call_count
            call_count += 1
            return evidence

        verifier._search_feature = fake_search

        result = asyncio.run(verifier.verify(features, accommodation_name="호텔A"))

        # max_queries=1 so only 1 search should have been made
        assert call_count == 1
        assert len(result) == 3


class TestVerifyToleratesSearchFailure:
    def test_verify_tolerates_search_failure(self) -> None:
        verifier = WebVerifier()

        feature = _make_verified(tag="노천탕", claim_level=ClaimLevel.PROBABLE)

        async def failing_search(name: str, tag: str) -> list[WebEvidence]:
            raise RuntimeError("network error")

        verifier._search_feature = failing_search

        result = asyncio.run(verifier.verify([feature], accommodation_name="호텔A"))

        # No crash; feature returned unchanged
        assert len(result) == 1
        assert result[0].tag == "노천탕"
        assert result[0].confidence == pytest.approx(feature.confidence)
        assert result[0].web_evidence == []
