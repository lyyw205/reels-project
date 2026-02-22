"""Claim gate — prevents exaggeration in marketing copy.

Assigns ClaimLevel tiers to features based on confidence scores,
controlling which expressions are permitted in generated copy.
"""

from __future__ import annotations

import logging
from typing import Any

from reels.production.models import (
    ClaimLevel,
    Feature,
    VerifiedFeature,
)

logger = logging.getLogger(__name__)

# Default thresholds
_CONFIRMED_THRESHOLD = 0.75
_PROBABLE_THRESHOLD = 0.50
_MULTI_EVIDENCE_BONUS = 0.10

# Factual claim words that require CONFIRMED level (Korean)
_FACTUAL_WORDS = frozenset(["무료", "제공", "무제한", "포함", "운영"])


class ClaimGate:
    """Evaluate feature confidence and assign claim levels."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("production", {}).get("claim_gate", {})
        self.confirmed_threshold = cfg.get("confirmed_threshold", _CONFIRMED_THRESHOLD)
        self.probable_threshold = cfg.get("probable_threshold", _PROBABLE_THRESHOLD)
        self.multi_evidence_bonus = cfg.get("multi_evidence_bonus", _MULTI_EVIDENCE_BONUS)
        self.factual_words = frozenset(cfg.get("factual_words", _FACTUAL_WORDS))

    def evaluate(self, features: list[Feature]) -> list[VerifiedFeature]:
        """Assign ClaimLevel to each feature based on confidence.

        Rules:
        1. confidence >= confirmed_threshold -> CONFIRMED
        2. probable_threshold <= confidence < confirmed_threshold -> PROBABLE
        3. confidence < probable_threshold -> SUGGESTIVE

        Bonus: +multi_evidence_bonus if evidence_images >= 2
        """
        verified = []
        for feature in features:
            adjusted = feature.confidence
            if len(feature.evidence_images) >= 2:
                adjusted = min(1.0, adjusted + self.multi_evidence_bonus)

            level = self._classify(adjusted)
            tone = self._level_to_tone(level)

            verified.append(
                VerifiedFeature(
                    **feature.model_dump(),
                    claim_level=level,
                    copy_tone=tone,
                )
            )

        # Sort by confidence descending
        verified.sort(key=lambda f: f.confidence, reverse=True)

        # Main feature rule: top 2 must be CONFIRMED for "main" usage
        # (This is enforced downstream by CopyWriter, but we log warnings here)
        for i, vf in enumerate(verified[:2]):
            if vf.claim_level != ClaimLevel.CONFIRMED:
                logger.warning(
                    "Main feature #%d '%s' is %s (not CONFIRMED) — "
                    "web verification recommended",
                    i + 1, vf.tag, vf.claim_level,
                )

        return verified

    def re_evaluate(self, features: list[VerifiedFeature]) -> list[VerifiedFeature]:
        """Re-evaluate ClaimLevel after WebVerifier updates confidence.

        Called after web verification to refresh stale claim levels.
        """
        result = []
        for f in features:
            adjusted = f.confidence
            if len(f.evidence_images) >= 2:
                adjusted = min(1.0, adjusted + self.multi_evidence_bonus)

            # Web evidence boosts confidence
            if f.web_evidence:
                avg_web_conf = sum(e.confidence for e in f.web_evidence) / len(f.web_evidence)
                adjusted = min(1.0, max(adjusted, avg_web_conf))

            new_level = self._classify(adjusted)
            new_tone = self._level_to_tone(new_level)

            result.append(
                f.model_copy(update={"claim_level": new_level, "copy_tone": new_tone})
            )
        result.sort(key=lambda f: f.confidence, reverse=True)
        return result

    def needs_web_verification(self, feature: VerifiedFeature, is_main: bool = False) -> bool:
        """Determine if a feature needs web verification.

        Trigger rules:
        1. Main feature (top 2) but not CONFIRMED
        2. Feature is PROBABLE and we want to promote it
        3. Never for SUGGESTIVE (too unreliable to verify)
        """
        if feature.claim_level == ClaimLevel.CONFIRMED:
            return False
        if feature.claim_level == ClaimLevel.SUGGESTIVE:
            return False  # Too uncertain — don't bother searching
        # PROBABLE features: verify if main
        return is_main

    def _classify(self, adjusted_confidence: float) -> ClaimLevel:
        if adjusted_confidence >= self.confirmed_threshold:
            return ClaimLevel.CONFIRMED
        if adjusted_confidence >= self.probable_threshold:
            return ClaimLevel.PROBABLE
        return ClaimLevel.SUGGESTIVE

    @staticmethod
    def _level_to_tone(level: ClaimLevel) -> str:
        return {
            ClaimLevel.CONFIRMED: "단정",
            ClaimLevel.PROBABLE: "암시",
            ClaimLevel.SUGGESTIVE: "분위기",
        }[level]
