"""Tests for reels.production.copy_writer module."""

from __future__ import annotations

from pathlib import Path

import pytest

from reels.production.copy_writer import CopyWriter
from reels.production.models import (
    AccommodationInput,
    ClaimLevel,
    FeatureCategory,
    TargetAudience,
    VerifiedFeature,
)


def _make_verified(
    tag: str = "노천탕",
    tag_en: str = "outdoor_bath",
    confidence: float = 0.80,
    claim_level: ClaimLevel = ClaimLevel.CONFIRMED,
    category: FeatureCategory = FeatureCategory.AMENITY,
    copy_tone: str = "단정",
) -> VerifiedFeature:
    return VerifiedFeature(
        tag=tag,
        tag_en=tag_en,
        confidence=confidence,
        claim_level=claim_level,
        category=category,
        copy_tone=copy_tone,
    )


def _make_context(
    name: str | None = "그랜드 호텔",
    region: str | None = "강원도",
    target_audience: TargetAudience = TargetAudience.COUPLE,
) -> AccommodationInput:
    return AccommodationInput(
        name=name,
        region=region,
        target_audience=target_audience,
        images=[Path("dummy.jpg")],
    )


class TestGenerateShotCount:
    def test_generate_returns_correct_shot_count(self) -> None:
        writer = CopyWriter()
        features = [_make_verified(tag=f"태그{i}") for i in range(5)]
        context = _make_context()
        result = writer.generate(features, context, shot_count=7)
        assert len(result) == 7

    def test_generate_returns_correct_shot_count_custom(self) -> None:
        writer = CopyWriter()
        features = [_make_verified()]
        context = _make_context()
        result = writer.generate(features, context, shot_count=3)
        assert len(result) == 3


class TestHookGeneration:
    def test_hook_confirmed_uses_tag(self) -> None:
        writer = CopyWriter()
        feature = _make_verified(tag="노천탕", claim_level=ClaimLevel.CONFIRMED)
        context = _make_context()
        result = writer.generate([feature], context, shot_count=1)
        assert result[0].hook_line is not None
        assert "노천탕" in result[0].hook_line

    def test_hook_suggestive_no_specific_tag(self) -> None:
        writer = CopyWriter()
        feature = _make_verified(tag="노천탕", claim_level=ClaimLevel.SUGGESTIVE)
        context = _make_context()
        result = writer.generate([feature], context, shot_count=1)
        assert result[0].hook_line is not None
        assert "노천탕" not in result[0].hook_line

    def test_hook_probable_no_specific_tag(self) -> None:
        writer = CopyWriter()
        feature = _make_verified(tag="사우나", claim_level=ClaimLevel.PROBABLE)
        context = _make_context()
        result = writer.generate([feature], context, shot_count=1)
        assert result[0].hook_line is not None
        assert "사우나" not in result[0].hook_line

    def test_hook_empty_features_fallback(self) -> None:
        writer = CopyWriter()
        context = _make_context()
        result = writer.generate([], context, shot_count=1)
        assert result[0].hook_line == "특별한 숙소"


class TestCaptionConstraints:
    def test_caption_max_chars(self) -> None:
        writer = CopyWriter()
        features = [_make_verified(tag="노천탕", claim_level=ClaimLevel.CONFIRMED)]
        context = _make_context()
        result = writer.generate(features, context, shot_count=2)
        # Shot 1 is a feature shot
        for line in result[1].caption_lines:
            assert len(line.text) <= writer.max_caption_chars

    def test_caption_max_lines(self) -> None:
        writer = CopyWriter()
        features = [_make_verified(tag="노천탕", claim_level=ClaimLevel.CONFIRMED)]
        context = _make_context()
        result = writer.generate(features, context, shot_count=2)
        assert len(result[1].caption_lines) <= writer.max_caption_lines

    def test_caption_max_chars_all_shots(self) -> None:
        writer = CopyWriter()
        features = [_make_verified(tag=f"태그{i}") for i in range(5)]
        context = _make_context()
        result = writer.generate(features, context, shot_count=7)
        for shot in result:
            for line in shot.caption_lines:
                assert len(line.text) <= writer.max_caption_chars

    def test_caption_max_lines_all_shots(self) -> None:
        writer = CopyWriter()
        features = [_make_verified(tag=f"태그{i}") for i in range(5)]
        context = _make_context()
        result = writer.generate(features, context, shot_count=7)
        for shot in result:
            assert len(shot.caption_lines) <= writer.max_caption_lines


class TestCheckFactualClaims:
    def test_check_factual_claims_confirmed_allows_all(self) -> None:
        writer = CopyWriter()
        violations = writer.check_factual_claims("무료 제공 무제한 포함 운영", ClaimLevel.CONFIRMED)
        assert violations == []

    def test_check_factual_claims_probable_catches_words(self) -> None:
        writer = CopyWriter()
        violations = writer.check_factual_claims("무료 조식", ClaimLevel.PROBABLE)
        assert "무료" in violations

    def test_check_factual_claims_suggestive_catches_words(self) -> None:
        writer = CopyWriter()
        violations = writer.check_factual_claims("제공 서비스", ClaimLevel.SUGGESTIVE)
        assert "제공" in violations

    def test_check_factual_claims_no_violation_clean_text(self) -> None:
        writer = CopyWriter()
        violations = writer.check_factual_claims("힐링 여행 특별한", ClaimLevel.PROBABLE)
        assert violations == []

    def test_check_factual_claims_multiple_violations(self) -> None:
        writer = CopyWriter()
        violations = writer.check_factual_claims("무료 제공 무제한", ClaimLevel.PROBABLE)
        assert len(violations) == 3
        assert set(violations) == {"무료", "제공", "무제한"}


class TestSanitizeFactual:
    def test_sanitize_replaces_factual(self) -> None:
        writer = CopyWriter()
        result = writer.sanitize_factual("무료 제공", ClaimLevel.PROBABLE)
        assert "무료" not in result
        assert "제공" not in result
        assert "특별한" in result
        assert "준비된" in result

    def test_sanitize_replaces_all_factual_words(self) -> None:
        writer = CopyWriter()
        text = "무료 제공 무제한 포함 운영"
        result = writer.sanitize_factual(text, ClaimLevel.SUGGESTIVE)
        for word in ["무료", "제공", "무제한", "포함", "운영"]:
            assert word not in result

    def test_sanitize_leaves_non_factual_unchanged(self) -> None:
        writer = CopyWriter()
        result = writer.sanitize_factual("힐링 여행", ClaimLevel.PROBABLE)
        assert result == "힐링 여행"


class TestCtaGeneration:
    def test_cta_uses_name(self) -> None:
        writer = CopyWriter()
        context = _make_context(name="오션뷰 리조트", region="제주도")
        result = writer.generate([], context, shot_count=1)
        # shot 0 with no features falls through to CTA logic via fallback hook
        # Test CTA directly via _generate_cta
        cta = writer._generate_cta(context)
        assert len(cta) == 1
        assert "오션뷰 리조트" in cta[0].text

    def test_cta_fallback_region(self) -> None:
        writer = CopyWriter()
        context = _make_context(name=None, region="제주도")
        cta = writer._generate_cta(context)
        assert len(cta) == 1
        assert "제주도" in cta[0].text

    def test_cta_fallback_no_name_no_region(self) -> None:
        writer = CopyWriter()
        context = _make_context(name=None, region=None)
        cta = writer._generate_cta(context)
        assert len(cta) == 1
        assert cta[0].text == "지금 예약하기"

    def test_cta_in_generated_shots(self) -> None:
        writer = CopyWriter()
        features = [_make_verified()]
        context = _make_context(name="힐링 펜션")
        # Shot 6 should be CTA
        result = writer.generate(features, context, shot_count=7)
        cta_shot = result[6]
        assert len(cta_shot.caption_lines) >= 1
        assert "힐링 펜션" in cta_shot.caption_lines[0].text


class TestVoScript:
    def test_vo_script_confirmed_uses_tag(self) -> None:
        writer = CopyWriter()
        features = [_make_verified(tag="노천탕", claim_level=ClaimLevel.CONFIRMED)]
        context = _make_context(name="힐링 리조트")
        script = writer.generate_vo_script(features, context)
        assert script is not None
        assert "노천탕" in script

    def test_vo_script_skips_suggestive(self) -> None:
        writer = CopyWriter()
        features = [
            _make_verified(tag="노천탕", claim_level=ClaimLevel.SUGGESTIVE),
            _make_verified(tag="수영장", claim_level=ClaimLevel.SUGGESTIVE),
        ]
        context = _make_context(name="힐링 리조트")
        script = writer.generate_vo_script(features, context)
        assert script is not None
        assert "노천탕" not in script
        assert "수영장" not in script

    def test_vo_script_probable_uses_tag(self) -> None:
        writer = CopyWriter()
        features = [_make_verified(tag="사우나", claim_level=ClaimLevel.PROBABLE)]
        context = _make_context(name="스파 호텔")
        script = writer.generate_vo_script(features, context)
        assert script is not None
        assert "사우나" in script

    def test_vo_script_empty_features_returns_none(self) -> None:
        writer = CopyWriter()
        context = _make_context()
        script = writer.generate_vo_script([], context)
        assert script is None

    def test_vo_script_uses_context_name(self) -> None:
        writer = CopyWriter()
        features = [_make_verified(claim_level=ClaimLevel.CONFIRMED)]
        context = _make_context(name="바다 리조트")
        script = writer.generate_vo_script(features, context)
        assert script is not None
        assert "바다 리조트" in script

    def test_vo_script_fallback_region(self) -> None:
        writer = CopyWriter()
        features = [_make_verified(claim_level=ClaimLevel.CONFIRMED)]
        context = _make_context(name=None, region="강원도")
        script = writer.generate_vo_script(features, context)
        assert script is not None
        assert "강원도" in script

    def test_vo_script_has_closing(self) -> None:
        writer = CopyWriter()
        features = [_make_verified(claim_level=ClaimLevel.CONFIRMED)]
        context = _make_context()
        script = writer.generate_vo_script(features, context)
        assert script is not None
        assert "지금 바로 확인해보세요" in script


class TestTrimToLength:
    def test_trim_to_length_short_text_unchanged(self) -> None:
        assert CopyWriter._trim_to_length("짧은", 10) == "짧은"

    def test_trim_to_length_exact_length_unchanged(self) -> None:
        text = "열두글자입니다이건"  # 9 chars
        assert CopyWriter._trim_to_length(text, 9) == text

    def test_trim_to_length_long_text_trimmed(self) -> None:
        text = "이것은매우긴텍스트입니다초과됩니다"
        result = CopyWriter._trim_to_length(text, 5)
        assert len(result) == 5
        assert result == text[:5]

    def test_trim_to_length_empty_string(self) -> None:
        assert CopyWriter._trim_to_length("", 10) == ""
