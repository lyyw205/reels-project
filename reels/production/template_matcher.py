"""Template matcher — selects best reference template from DB."""

from __future__ import annotations

import logging
from typing import Any

from reels.db.repository import TemplateRepository
from reels.models.template import CameraType, Template
from reels.production.models import (
    AccommodationInput,
    MatchResult,
    TargetAudience,
    VerifiedFeature,
)
from reels.storage import TemplateArchiver

logger = logging.getLogger(__name__)

# Default fallback structure when no template matches
DEFAULT_STRUCTURE = [
    {"role": "hook",    "duration": 1.2, "camera": CameraType.PUSH_IN},
    {"role": "feature", "duration": 2.3, "camera": CameraType.STATIC},
    {"role": "feature", "duration": 2.0, "camera": CameraType.PAN_RIGHT},
    {"role": "feature", "duration": 2.0, "camera": CameraType.STATIC},
    {"role": "support", "duration": 1.5, "camera": CameraType.GIMBAL_SMOOTH},
    {"role": "support", "duration": 1.5, "camera": CameraType.PAN_LEFT},
    {"role": "cta",     "duration": 1.5, "camera": CameraType.STATIC},
]

# BPM ranges by target audience mood
_AUDIENCE_BPM = {
    TargetAudience.COUPLE: (70, 100),    # Relaxed, romantic
    TargetAudience.FAMILY: (90, 120),    # Upbeat, energetic
    TargetAudience.SOLO: (60, 90),       # Calm, meditative
    TargetAudience.FRIENDS: (100, 130),  # Fun, dynamic
}


class TemplateMatcher:
    """Select best reference template for production."""

    def __init__(
        self,
        repo: TemplateRepository,
        archiver: TemplateArchiver | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.repo = repo
        self.archiver = archiver
        cfg = (config or {}).get("production", {}).get("template_matching", {})
        self.duration_range = tuple(cfg.get("duration_range", [10.0, 15.0]))
        self.target_shot_count = tuple(cfg.get("target_shot_count", [6, 8]))
        self.weights = cfg.get("weights", {
            "place_overlap": 0.30,
            "duration_fit": 0.20,
            "camera_variety": 0.15,
            "rhythm_match": 0.15,
            "shot_count_fit": 0.20,
        })

    def find_best(
        self,
        features: list[VerifiedFeature],
        context: AccommodationInput,
        duration_range: tuple[float, float] | None = None,
    ) -> MatchResult | None:
        """Find best matching template. Returns None if no good match found."""
        dr = duration_range or self.duration_range

        # Get candidate templates from DB
        candidates = self.repo.search_by_duration(
            min_sec=max(0, dr[0] - 5),  # Wider range for initial filter
            max_sec=dr[1] + 5,
        )

        if not candidates:
            logger.info("No template candidates found in DB")
            return None

        # Score each candidate
        scored = []
        for template in candidates:
            score = self._score_template(template, features, context, dr)
            scored.append((template, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        best_template, best_score = scored[0]

        # Minimum score threshold
        if best_score < 0.3:
            logger.info(
                "Best template %s scored %.2f (below 0.3 threshold)",
                best_template.template_id, best_score,
            )
            return None

        logger.info(
            "Matched template: %s (score=%.2f, shots=%d, %.1fs)",
            best_template.template_id, best_score,
            best_template.shot_count, best_template.total_duration_sec,
        )

        return MatchResult(
            template_id=best_template.template_id,
            score=round(best_score, 3),
            shot_count=best_template.shot_count,
            template_duration=best_template.total_duration_sec,
        )

    def _score_template(
        self,
        template: Template,
        features: list[VerifiedFeature],
        context: AccommodationInput,
        duration_range: tuple[float, float],
    ) -> float:
        """Compute composite match score (0-1)."""
        w = self.weights

        place_score = self._score_place_overlap(template, features)
        duration_score = self._score_duration_fit(template, duration_range)
        camera_score = self._score_camera_variety(template)
        rhythm_score = self._score_rhythm_match(template, context)
        shot_score = self._score_shot_count_fit(template)

        total = (
            w.get("place_overlap", 0.30) * place_score
            + w.get("duration_fit", 0.20) * duration_score
            + w.get("camera_variety", 0.15) * camera_score
            + w.get("rhythm_match", 0.15) * rhythm_score
            + w.get("shot_count_fit", 0.20) * shot_score
        )

        return total

    def _score_place_overlap(
        self, template: Template, features: list[VerifiedFeature],
    ) -> float:
        """How well do template shot places match our features?"""
        if not features or not template.shots:
            return 0.0

        feature_tags = {f.tag_en.lower() for f in features}
        template_places = {s.place_label.lower() for s in template.shots}

        if not feature_tags or not template_places:
            return 0.0

        overlap = feature_tags & template_places
        # Jaccard similarity
        union = feature_tags | template_places
        return len(overlap) / len(union) if union else 0.0

    def _score_duration_fit(
        self, template: Template, duration_range: tuple[float, float],
    ) -> float:
        """How well does template duration fit the target range?"""
        d = template.total_duration_sec
        lo, hi = duration_range

        if lo <= d <= hi:
            return 1.0
        # Gradual penalty for being outside range
        if d < lo:
            return max(0.0, 1.0 - (lo - d) / lo)
        return max(0.0, 1.0 - (d - hi) / hi)

    def _score_camera_variety(self, template: Template) -> float:
        """Reward templates with diverse camera types."""
        if not template.shots:
            return 0.0
        types = {str(s.camera.type) for s in template.shots}
        # More variety is better, up to 4 types
        return min(1.0, len(types) / 4.0)

    def _score_rhythm_match(
        self, template: Template, context: AccommodationInput,
    ) -> float:
        """How well does BPM match the target audience mood?"""
        bpms = [s.bpm for s in template.shots if s.bpm and s.bpm > 0]
        if not bpms:
            return 0.5  # Neutral if no BPM data

        avg_bpm = sum(bpms) / len(bpms)
        target_lo, target_hi = _AUDIENCE_BPM.get(
            context.target_audience, (70, 120),
        )

        if target_lo <= avg_bpm <= target_hi:
            return 1.0
        # Gradual penalty
        if avg_bpm < target_lo:
            return max(0.0, 1.0 - (target_lo - avg_bpm) / 50)
        return max(0.0, 1.0 - (avg_bpm - target_hi) / 50)

    def _score_shot_count_fit(self, template: Template) -> float:
        """How close is shot count to target 6-8 range?"""
        n = template.shot_count
        lo, hi = self.target_shot_count

        if lo <= n <= hi:
            return 1.0
        if n < lo:
            return max(0.0, 1.0 - (lo - n) / lo)
        return max(0.0, 1.0 - (n - hi) / hi)

    def get_fallback_structure(self) -> list[dict]:
        """Return default shot structure when no template matches."""
        return [dict(s) for s in DEFAULT_STRUCTURE]

    def default_shot_count(self) -> int:
        """Return midpoint of target shot count range."""
        lo, hi = self.target_shot_count
        return (lo + hi) // 2
