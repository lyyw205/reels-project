"""Copy writer — generates marketing copy for shorts with tone control."""

from __future__ import annotations

import logging
import re
from typing import Any

from reels.production.models import (
    CaptionLine,
    ClaimLevel,
    ShotCopy,
    VerifiedFeature,
    AccommodationInput,
    TargetAudience,
)

logger = logging.getLogger(__name__)

_MAX_CAPTION_CHARS = 14
_MAX_CAPTION_LINES = 2
_HOOK_MIN_CHARS = 7
_HOOK_MAX_CHARS = 12

# Words that require CONFIRMED level
_FACTUAL_WORDS = frozenset(["무료", "제공", "무제한", "포함", "운영"])

# Tone templates by ClaimLevel
_TONE_TEMPLATES = {
    ClaimLevel.CONFIRMED: {
        "patterns": ["{tag} 있는 {benefit}", "{tag} 즐기는 {benefit}", "{tag} 제공"],
        "allowed_verbs": ["있는", "제공", "즐기는", "만나는"],
    },
    ClaimLevel.PROBABLE: {
        "patterns": ["{tag} 느낌의 {benefit}", "{tag} 같은 {benefit}"],
        "forbidden_verbs": ["있다", "제공한다", "즐길 수 있다"],
    },
    ClaimLevel.SUGGESTIVE: {
        "patterns": ["특별한 {benefit}", "여유로운 {benefit}"],
        "forbidden": ["specific_tag_names"],  # Cannot use specific amenity names
    },
}

# Audience-specific benefit words
_AUDIENCE_BENEFITS = {
    TargetAudience.COUPLE: ["프라이빗", "로맨틱", "힐링", "특별한"],
    TargetAudience.FAMILY: ["함께하는", "즐거운", "편안한", "안심"],
    TargetAudience.SOLO: ["나만의", "조용한", "여유로운", "힐링"],
    TargetAudience.FRIENDS: ["신나는", "함께", "특별한", "인생샷"],
}


class CopyWriter:
    """Generate marketing copy with tone rules based on ClaimLevel."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("production", {}).get("copy", {})
        self.max_caption_chars = cfg.get("max_caption_chars", _MAX_CAPTION_CHARS)
        self.max_caption_lines = cfg.get("max_caption_lines", _MAX_CAPTION_LINES)
        self.hook_min_chars = cfg.get("hook_min_chars", _HOOK_MIN_CHARS)
        self.hook_max_chars = cfg.get("hook_max_chars", _HOOK_MAX_CHARS)
        self.factual_words = frozenset(cfg.get("factual_words", _FACTUAL_WORDS))

    def generate(
        self,
        features: list[VerifiedFeature],
        context: AccommodationInput,
        shot_count: int = 7,
    ) -> list[ShotCopy]:
        """Generate copy for each shot based on features and roles.

        Shot allocation:
        - Shot 0: HOOK (best visual feature)
        - Shots 1-3: FEATURE (top 3 features)
        - Shots 4-5: SUPPORT (secondary features)
        - Shots 6+: CTA
        """
        copies = []

        for i in range(shot_count):
            if i == 0:
                # Hook shot
                top = features[0] if features else None
                hook = self._generate_hook(top, context) if top else "특별한 숙소"
                copies.append(ShotCopy(hook_line=hook))
            elif i <= 3 and i - 1 < len(features):
                # Feature shots
                feat = features[i - 1]
                captions = self._generate_captions(feat, context)
                copies.append(ShotCopy(caption_lines=captions))
            elif i <= 5 and i - 1 < len(features):
                # Support shots
                feat_idx = min(i - 1, len(features) - 1)
                feat = features[feat_idx]
                captions = self._generate_captions(feat, context)
                copies.append(ShotCopy(caption_lines=captions))
            else:
                # CTA shot
                cta = self._generate_cta(context)
                copies.append(ShotCopy(caption_lines=cta))

        return copies

    def _generate_hook(
        self, feature: VerifiedFeature, context: AccommodationInput,
    ) -> str:
        """Generate hook line (7-12 Korean characters)."""
        benefits = _AUDIENCE_BENEFITS.get(context.target_audience, ["특별한"])
        benefit = benefits[0]

        if feature.claim_level == ClaimLevel.CONFIRMED:
            hook = f"{feature.tag} {benefit} 숙소"
        elif feature.claim_level == ClaimLevel.PROBABLE:
            hook = f"{benefit} 감성 숙소"
        else:
            hook = f"{benefit} 힐링 여행"

        # Trim to max length
        hook = self._trim_to_length(hook, self.hook_max_chars)
        return hook

    def _generate_captions(
        self, feature: VerifiedFeature, context: AccommodationInput,
    ) -> list[CaptionLine]:
        """Generate caption lines for a feature (max 2 lines, 12-14 chars each)."""
        lines = []
        benefits = _AUDIENCE_BENEFITS.get(context.target_audience, ["특별한"])

        if feature.claim_level == ClaimLevel.CONFIRMED:
            line1 = f"{feature.tag}"
            line2 = f"{benefits[0]} 경험"
        elif feature.claim_level == ClaimLevel.PROBABLE:
            line1 = f"{feature.tag} 느낌의"
            line2 = f"{benefits[0]} 공간"
        else:
            line1 = f"특별한 공간에서"
            line2 = f"{benefits[0]} 시간"

        line1 = self._trim_to_length(line1, self.max_caption_chars)
        line2 = self._trim_to_length(line2, self.max_caption_chars)

        lines.append(CaptionLine(text=line1, position="bottom_center"))
        if line2:
            lines.append(CaptionLine(text=line2, position="bottom_center"))

        # Validate: no factual claims in non-CONFIRMED
        validated = []
        for line in lines[:self.max_caption_lines]:
            warnings = self.check_factual_claims(line.text, feature.claim_level)
            if warnings:
                logger.warning("Factual claim in %s copy: %s", feature.claim_level, warnings)
                line = CaptionLine(
                    text=self.sanitize_factual(line.text, feature.claim_level),
                    start_sec=line.start_sec,
                    end_sec=line.end_sec,
                    position=line.position,
                )
            validated.append(line)

        return validated

    def _generate_cta(self, context: AccommodationInput) -> list[CaptionLine]:
        """Generate call-to-action caption."""
        if context.name:
            cta = f"{context.name}"
        elif context.region:
            cta = f"{context.region} 숙소"
        else:
            cta = "지금 예약하기"

        cta = self._trim_to_length(cta, self.max_caption_chars)
        return [CaptionLine(text=cta, position="bottom_center")]

    def check_factual_claims(self, text: str, claim_level: ClaimLevel) -> list[str]:
        """Check for factual claim words that aren't allowed at this ClaimLevel.

        Factual words (무료, 제공, 무제한, 포함, 운영) are only allowed
        at CONFIRMED level. Returns list of violating words found.
        """
        if claim_level == ClaimLevel.CONFIRMED:
            return []

        violations = []
        for word in self.factual_words:
            if word in text:
                violations.append(word)
        return violations

    def sanitize_factual(self, text: str, claim_level: ClaimLevel) -> str:
        """Remove/replace factual claim words for non-CONFIRMED features."""
        result = text
        replacements = {
            "무료": "특별한",
            "제공": "준비된",
            "무제한": "여유로운",
            "포함": "함께하는",
            "운영": "만나는",
        }
        for word, replacement in replacements.items():
            result = result.replace(word, replacement)
        return result

    @staticmethod
    def _trim_to_length(text: str, max_chars: int) -> str:
        """Trim Korean text to max character count."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def generate_vo_script(
        self, features: list[VerifiedFeature], context: AccommodationInput,
    ) -> str | None:
        """Generate voiceover narration script (optional)."""
        if not features:
            return None

        parts = []
        name = context.name or context.region or "이곳"

        # Opening
        parts.append(f"{name}에서의 특별한 하루.")

        # Feature highlights (top 3)
        for feat in features[:3]:
            if feat.claim_level == ClaimLevel.CONFIRMED:
                parts.append(f"{feat.tag}까지 즐길 수 있는 곳.")
            elif feat.claim_level == ClaimLevel.PROBABLE:
                parts.append(f"{feat.tag} 느낌의 특별한 공간.")
            # Skip SUGGESTIVE for VO

        # Closing
        parts.append("지금 바로 확인해보세요.")

        return " ".join(parts)
