"""Tests for reels.production.claim_gate module."""

from __future__ import annotations

import pytest

from reels.production.claim_gate import ClaimGate
from reels.production.models import (
    ClaimLevel,
    Feature,
    FeatureCategory,
    VerifiedFeature,
    WebEvidence,
)


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


def _make_verified(
    tag: str = "노천탕",
    tag_en: str = "outdoor_bath",
    confidence: float = 0.80,
    evidence_images: list[str] | None = None,
    claim_level: ClaimLevel = ClaimLevel.CONFIRMED,
    web_evidence: list[WebEvidence] | None = None,
    copy_tone: str = "단정",
) -> VerifiedFeature:
    return VerifiedFeature(
        tag=tag,
        tag_en=tag_en,
        confidence=confidence,
        evidence_images=evidence_images or [],
        claim_level=claim_level,
        web_evidence=web_evidence or [],
        copy_tone=copy_tone,
    )


class TestClaimGateEvaluate:
    def test_high_confidence_is_confirmed(self) -> None:
        gate = ClaimGate()
        features = [_make_feature(confidence=0.85)]
        result = gate.evaluate(features)
        assert len(result) == 1
        assert result[0].claim_level == ClaimLevel.CONFIRMED

    def test_mid_confidence_is_probable(self) -> None:
        gate = ClaimGate()
        features = [_make_feature(confidence=0.60)]
        result = gate.evaluate(features)
        assert result[0].claim_level == ClaimLevel.PROBABLE

    def test_low_confidence_is_suggestive(self) -> None:
        gate = ClaimGate()
        features = [_make_feature(confidence=0.30)]
        result = gate.evaluate(features)
        assert result[0].claim_level == ClaimLevel.SUGGESTIVE

    def test_multi_evidence_bonus(self) -> None:
        # 0.70 + 0.10 bonus = 0.80 >= 0.75 -> CONFIRMED
        gate = ClaimGate()
        features = [_make_feature(confidence=0.70, evidence_images=["a.jpg", "b.jpg"])]
        result = gate.evaluate(features)
        assert result[0].claim_level == ClaimLevel.CONFIRMED

    def test_multi_evidence_bonus_caps_at_1(self) -> None:
        # 0.95 + 0.10 would exceed 1.0 — must be capped at 1.0
        gate = ClaimGate()
        features = [_make_feature(confidence=0.95, evidence_images=["a.jpg", "b.jpg", "c.jpg"])]
        result = gate.evaluate(features)
        assert result[0].claim_level == ClaimLevel.CONFIRMED
        # confidence on the model is the original value (not adjusted)
        assert result[0].confidence == pytest.approx(0.95)

    def test_features_sorted_by_confidence(self) -> None:
        gate = ClaimGate()
        features = [
            _make_feature(tag="낮은", confidence=0.30),
            _make_feature(tag="높은", confidence=0.90),
            _make_feature(tag="중간", confidence=0.60),
        ]
        result = gate.evaluate(features)
        confidences = [f.confidence for f in result]
        assert confidences == sorted(confidences, reverse=True)
        assert result[0].tag == "높은"
        assert result[-1].tag == "낮은"


class TestClaimGateReEvaluate:
    def test_re_evaluate_after_web_evidence(self) -> None:
        # Feature at 0.60 (PROBABLE) gets web_evidence at 0.90 -> adjusted to 0.90 -> CONFIRMED
        gate = ClaimGate()
        vf = _make_verified(
            confidence=0.60,
            claim_level=ClaimLevel.PROBABLE,
            web_evidence=[WebEvidence(claim="노천탕 있음", url="http://example.com", snippet="노천탕", confidence=0.90)],
            copy_tone="암시",
        )
        result = gate.re_evaluate([vf])
        assert result[0].claim_level == ClaimLevel.CONFIRMED

    def test_re_evaluate_preserves_existing_confirmed(self) -> None:
        gate = ClaimGate()
        vf = _make_verified(confidence=0.85, claim_level=ClaimLevel.CONFIRMED, copy_tone="단정")
        result = gate.re_evaluate([vf])
        assert result[0].claim_level == ClaimLevel.CONFIRMED


class TestNeedsWebVerification:
    def test_needs_web_verification_main_probable(self) -> None:
        gate = ClaimGate()
        vf = _make_verified(confidence=0.60, claim_level=ClaimLevel.PROBABLE, copy_tone="암시")
        assert gate.needs_web_verification(vf, is_main=True) is True

    def test_needs_web_verification_confirmed(self) -> None:
        gate = ClaimGate()
        vf = _make_verified(confidence=0.85, claim_level=ClaimLevel.CONFIRMED, copy_tone="단정")
        assert gate.needs_web_verification(vf, is_main=True) is False

    def test_needs_web_verification_suggestive(self) -> None:
        gate = ClaimGate()
        vf = _make_verified(confidence=0.30, claim_level=ClaimLevel.SUGGESTIVE, copy_tone="분위기")
        # SUGGESTIVE is too uncertain — never verify
        assert gate.needs_web_verification(vf, is_main=True) is False


class TestToneMapping:
    def test_tone_mapping(self) -> None:
        gate = ClaimGate()
        features = [
            _make_feature(tag="확정", confidence=0.85),
            _make_feature(tag="가능성", confidence=0.60),
            _make_feature(tag="분위기", confidence=0.30),
        ]
        result = gate.evaluate(features)
        by_tag = {f.tag: f for f in result}
        assert by_tag["확정"].copy_tone == "단정"
        assert by_tag["가능성"].copy_tone == "암시"
        assert by_tag["분위기"].copy_tone == "분위기"


class TestCustomThresholds:
    def test_custom_thresholds(self) -> None:
        # Raise confirmed threshold to 0.90, probable to 0.70
        config = {
            "production": {
                "claim_gate": {
                    "confirmed_threshold": 0.90,
                    "probable_threshold": 0.70,
                }
            }
        }
        gate = ClaimGate(config=config)
        # 0.85 is normally CONFIRMED but with threshold 0.90 -> PROBABLE
        features = [_make_feature(confidence=0.85)]
        result = gate.evaluate(features)
        assert result[0].claim_level == ClaimLevel.PROBABLE

        # 0.60 is normally PROBABLE but with threshold 0.70 -> SUGGESTIVE
        features2 = [_make_feature(confidence=0.60)]
        result2 = gate.evaluate(features2)
        assert result2[0].claim_level == ClaimLevel.SUGGESTIVE
