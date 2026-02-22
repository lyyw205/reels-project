"""Storyboard builder — assembles shots from features, copies, and template structure."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from reels.models.template import CameraType, EditInfo, Template, TransitionType
from reels.production.models import (
    AccommodationInput,
    ShotCopy,
    ShotRole,
    Storyboard,
    StoryboardShot,
    VerifiedFeature,
)
from reels.production.template_matcher import DEFAULT_STRUCTURE

logger = logging.getLogger(__name__)

# Default timing for each role (seconds)
_DEFAULT_TIMING = {
    ShotRole.HOOK: 1.2,
    ShotRole.FEATURE: 2.0,
    ShotRole.SUPPORT: 1.5,
    ShotRole.CTA: 1.5,
}


class StoryboardBuilder:
    """Build storyboard from features + copies + template structure."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("production", {}).get("render", {})
        self.safe_area_margin = cfg.get("safe_area_margin", 0.15)
        self.ken_burns_default = cfg.get("ken_burns_default", True)

    def build(
        self,
        features: list[VerifiedFeature],
        copies: list[ShotCopy],
        template: Template | None,
        assets: list[Path],
        context: AccommodationInput,
    ) -> Storyboard:
        """Assemble final storyboard."""
        # Determine shot structure from template or fallback
        if template and template.shots:
            structure = self._extract_template_structure(template)
        else:
            structure = [dict(s) for s in DEFAULT_STRUCTURE]

        shot_count = len(structure)

        # Assign roles
        roles = self._assign_roles(features, shot_count)

        # Map assets to shots
        asset_map = self._match_assets_to_shots(features, assets, roles)

        # Build timing
        timings = self._build_timing(structure, template)

        # Assemble shots
        shots = []
        for i in range(shot_count):
            role = roles[i]
            start_sec, end_sec = timings[i]
            duration = round(end_sec - start_sec, 3)

            # Get feature for this shot
            feature = self._get_feature_for_role(features, role, i)

            # Get copy
            copy = copies[i] if i < len(copies) else ShotCopy()

            # Get camera suggestion from template or fallback
            camera = self._get_camera(structure, i)

            # Get asset
            asset_path = asset_map.get(i, "")

            shot = StoryboardShot(
                shot_id=i,
                role=role,
                start_sec=round(start_sec, 3),
                end_sec=round(end_sec, 3),
                duration_sec=duration,
                asset_type="video_clip" if self._is_video(asset_path) else "image",
                asset_path=str(asset_path) if asset_path else "",
                feature_tag=feature.tag_en if feature else None,
                claim_level=feature.claim_level if feature else None,
                place_label=feature.tag_en if feature else "other",
                camera_suggestion=camera,
                shot_copy=copy,
                edit=EditInfo(
                    speed=1.0,
                    transition_out=(
                        TransitionType.CUT_ON_BEAT
                        if role in (ShotRole.HOOK, ShotRole.FEATURE)
                        else TransitionType.CUT
                    ),
                ),
            )
            shots.append(shot)

        total_duration = round(sum(s.duration_sec for s in shots), 3)

        storyboard = Storyboard(
            project_id=self._generate_project_id(),
            accommodation_name=context.name,
            target_audience=context.target_audience,
            features=features,
            template_ref=template.template_id if template else None,
            total_duration_sec=total_duration,
            shots=shots,
        )

        logger.info(
            "Built storyboard: %d shots, %.1fs, template=%s",
            len(shots), total_duration, storyboard.template_ref,
        )
        return storyboard

    def _assign_roles(
        self, features: list[VerifiedFeature], shot_count: int,
    ) -> list[ShotRole]:
        """Assign roles to shots: HOOK → FEATURE → SUPPORT → CTA."""
        roles = []
        feature_count = min(3, len(features))  # Max 3 feature shots
        support_count = min(2, max(0, len(features) - 3))  # Up to 2 support

        for i in range(shot_count):
            if i == 0:
                roles.append(ShotRole.HOOK)
            elif i <= feature_count:
                roles.append(ShotRole.FEATURE)
            elif i <= feature_count + support_count:
                roles.append(ShotRole.SUPPORT)
            else:
                roles.append(ShotRole.CTA)

        return roles

    def _match_assets_to_shots(
        self,
        features: list[VerifiedFeature],
        assets: list[Path],
        roles: list[ShotRole],
    ) -> dict[int, Path]:
        """Map assets to shots based on feature evidence images."""
        asset_map: dict[int, Path] = {}
        used_assets: set[int] = set()

        # Build a lookup: image filename → asset path
        asset_by_name = {a.name: a for a in assets}

        # First pass: match by evidence images
        feat_idx = 0
        for i, role in enumerate(roles):
            if role in (ShotRole.FEATURE, ShotRole.SUPPORT, ShotRole.HOOK):
                if feat_idx < len(features):
                    feat = features[min(feat_idx, len(features) - 1)]
                    for img_name in feat.evidence_images:
                        if img_name in asset_by_name:
                            asset_idx = assets.index(asset_by_name[img_name])
                            if asset_idx not in used_assets:
                                asset_map[i] = assets[asset_idx]
                                used_assets.add(asset_idx)
                                break
                if role != ShotRole.HOOK:
                    feat_idx += 1

        # Second pass: fill remaining with unused assets
        unused = [j for j in range(len(assets)) if j not in used_assets]
        unused_idx = 0
        for i in range(len(roles)):
            if i not in asset_map and unused_idx < len(unused):
                asset_map[i] = assets[unused[unused_idx]]
                unused_idx += 1

        return asset_map

    def _extract_template_structure(self, template: Template) -> list[dict]:
        """Extract shot structure from reference template."""
        structure = []
        for shot in template.shots:
            structure.append({
                "role": "feature",  # Will be re-assigned
                "duration": shot.duration_sec,
                "camera": shot.camera.type,
            })
        return structure

    def _build_timing(
        self, structure: list[dict], template: Template | None,
    ) -> list[tuple[float, float]]:
        """Build shot timings from structure."""
        timings = []
        current = 0.0

        for s in structure:
            duration = s.get("duration", 1.5)
            timings.append((current, current + duration))
            current += duration

        return timings

    def _get_feature_for_role(
        self, features: list[VerifiedFeature], role: ShotRole, shot_idx: int,
    ) -> VerifiedFeature | None:
        """Get the appropriate feature for a given role and position."""
        if not features:
            return None
        if role == ShotRole.HOOK:
            return features[0]  # Best feature for hook
        if role == ShotRole.FEATURE:
            idx = shot_idx - 1  # Shot 1 → feature[0], Shot 2 → feature[1]
            return features[idx] if idx < len(features) else None
        if role == ShotRole.SUPPORT:
            idx = shot_idx - 1
            return features[idx] if idx < len(features) else features[-1]
        return None  # CTA doesn't need a feature

    @staticmethod
    def _get_camera(structure: list[dict], index: int) -> CameraType:
        """Get camera type from structure."""
        if index < len(structure):
            cam = structure[index].get("camera", CameraType.STATIC)
            if isinstance(cam, CameraType):
                return cam
            try:
                return CameraType(str(cam))
            except ValueError:
                return CameraType.STATIC
        return CameraType.STATIC

    @staticmethod
    def _is_video(path: str | Path) -> bool:
        """Check if path is a video file."""
        if not path:
            return False
        suffix = Path(str(path)).suffix.lower()
        return suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    @staticmethod
    def _generate_project_id() -> str:
        """Generate unique project ID."""
        import uuid
        return f"shorts_{uuid.uuid4().hex[:8]}"
