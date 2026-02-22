"""Tests for reels.production.models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reels.production.models import (
    AccommodationInput,
    AssetMapping,
    AssetTransform,
    CaptionLine,
    ClaimLevel,
    Feature,
    FeatureCategory,
    MatchResult,
    MusicSpec,
    PriceRange,
    ProductionResult,
    RenderSpec,
    ShotCopy,
    ShotRole,
    Storyboard,
    StoryboardShot,
    TargetAudience,
    VerifiedFeature,
    WebEvidence,
)
from reels.models.template import CameraType, EditInfo, TemplateFormat, TransitionType


# ─── Enums ───────────────────────────────────────────────────────

class TestTargetAudience:
    def test_values(self) -> None:
        assert TargetAudience.COUPLE == "couple"
        assert TargetAudience.FAMILY == "family"
        assert TargetAudience.SOLO == "solo"
        assert TargetAudience.FRIENDS == "friends"


class TestPriceRange:
    def test_values(self) -> None:
        assert PriceRange.BUDGET == "budget"
        assert PriceRange.MID == "mid"
        assert PriceRange.LUXURY == "luxury"


class TestFeatureCategory:
    def test_values(self) -> None:
        assert FeatureCategory.SCENE == "scene"
        assert FeatureCategory.AMENITY == "amenity"
        assert FeatureCategory.VIEW == "view"
        assert FeatureCategory.DINING == "dining"
        assert FeatureCategory.ACTIVITY == "activity"


class TestClaimLevel:
    def test_values(self) -> None:
        assert ClaimLevel.CONFIRMED == "confirmed"
        assert ClaimLevel.PROBABLE == "probable"
        assert ClaimLevel.SUGGESTIVE == "suggestive"


class TestShotRole:
    def test_values(self) -> None:
        assert ShotRole.HOOK == "hook"
        assert ShotRole.FEATURE == "feature"
        assert ShotRole.SUPPORT == "support"
        assert ShotRole.CTA == "cta"


# ─── AccommodationInput ───────────────────────────────────────────

class TestAccommodationInput:
    def test_requires_images(self) -> None:
        with pytest.raises(ValidationError):
            AccommodationInput()  # missing images

    def test_minimal_with_images(self) -> None:
        ai = AccommodationInput(images=[Path("/tmp/a.jpg")])
        assert ai.target_audience == TargetAudience.COUPLE
        assert ai.video_clips == []
        assert ai.name is None

    def test_full(self) -> None:
        ai = AccommodationInput(
            name="Grand Hotel",
            region="Jeju",
            target_audience=TargetAudience.FAMILY,
            price_range=PriceRange.LUXURY,
            images=[Path("/tmp/a.jpg"), Path("/tmp/b.jpg")],
            video_clips=[Path("/tmp/clip.mp4")],
            custom_instructions="Focus on the pool.",
        )
        assert ai.name == "Grand Hotel"
        assert ai.price_range == PriceRange.LUXURY
        assert len(ai.images) == 2


# ─── Feature ─────────────────────────────────────────────────────

class TestFeature:
    def test_confidence_valid(self) -> None:
        f = Feature(tag="노천탕", tag_en="outdoor_bath", confidence=0.8)
        assert f.confidence == 0.8
        assert f.category == FeatureCategory.SCENE

    def test_confidence_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Feature(tag="노천탕", tag_en="outdoor_bath", confidence=-0.1)

    def test_confidence_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Feature(tag="노천탕", tag_en="outdoor_bath", confidence=1.1)

    def test_confidence_boundary_values(self) -> None:
        f0 = Feature(tag="a", tag_en="a", confidence=0.0)
        f1 = Feature(tag="a", tag_en="a", confidence=1.0)
        assert f0.confidence == 0.0
        assert f1.confidence == 1.0

    def test_defaults(self) -> None:
        f = Feature(tag="오션뷰", tag_en="ocean_view", confidence=0.5)
        assert f.evidence_images == []
        assert f.description == ""
        assert f.category == FeatureCategory.SCENE


# ─── VerifiedFeature ─────────────────────────────────────────────

class TestVerifiedFeature:
    def test_inherits_from_feature(self) -> None:
        vf = VerifiedFeature(tag="노천탕", tag_en="outdoor_bath", confidence=0.9)
        assert isinstance(vf, Feature)

    def test_defaults(self) -> None:
        vf = VerifiedFeature(tag="수영장", tag_en="pool", confidence=0.6)
        assert vf.claim_level == ClaimLevel.SUGGESTIVE
        assert vf.web_evidence == []
        assert vf.copy_tone == "분위기"

    def test_with_web_evidence(self) -> None:
        ev = WebEvidence(
            claim="노천탕 있음",
            url="https://example.com",
            snippet="야외 노천탕이 있습니다.",
            confidence=0.9,
        )
        vf = VerifiedFeature(
            tag="노천탕",
            tag_en="outdoor_bath",
            confidence=0.85,
            claim_level=ClaimLevel.CONFIRMED,
            web_evidence=[ev],
        )
        assert vf.claim_level == ClaimLevel.CONFIRMED
        assert len(vf.web_evidence) == 1
        assert vf.web_evidence[0].confidence == 0.9

    def test_web_evidence_confidence_range(self) -> None:
        with pytest.raises(ValidationError):
            WebEvidence(claim="x", url="http://x.com", snippet="x", confidence=1.5)


# ─── StoryboardShot ───────────────────────────────────────────────

class TestStoryboardShot:
    def test_defaults(self) -> None:
        shot = StoryboardShot(
            shot_id=0,
            role=ShotRole.HOOK,
            start_sec=0.0,
            end_sec=3.0,
            duration_sec=3.0,
        )
        assert shot.asset_type == "image"
        assert shot.asset_path == ""
        assert shot.feature_tag is None
        assert shot.claim_level is None
        assert shot.place_label == "other"
        assert shot.camera_suggestion == CameraType.STATIC
        assert isinstance(shot.copy, ShotCopy)
        assert isinstance(shot.edit, EditInfo)

    def test_edit_default_transition(self) -> None:
        shot = StoryboardShot(
            shot_id=1,
            role=ShotRole.FEATURE,
            start_sec=3.0,
            end_sec=6.0,
            duration_sec=3.0,
        )
        assert shot.edit.transition_out == TransitionType.CUT

    def test_with_feature(self) -> None:
        shot = StoryboardShot(
            shot_id=2,
            role=ShotRole.FEATURE,
            start_sec=3.0,
            end_sec=6.0,
            duration_sec=3.0,
            feature_tag="노천탕",
            claim_level=ClaimLevel.CONFIRMED,
            camera_suggestion=CameraType.GIMBAL_SMOOTH,
        )
        assert shot.feature_tag == "노천탕"
        assert shot.claim_level == ClaimLevel.CONFIRMED


# ─── Storyboard ──────────────────────────────────────────────────

class TestStoryboard:
    def _make_shot(self, shot_id: int, start: float, end: float) -> StoryboardShot:
        return StoryboardShot(
            shot_id=shot_id,
            role=ShotRole.FEATURE,
            start_sec=start,
            end_sec=end,
            duration_sec=end - start,
        )

    def test_defaults(self) -> None:
        sb = Storyboard(project_id="proj-001")
        assert sb.accommodation_name is None
        assert sb.target_audience == TargetAudience.COUPLE
        assert sb.features == []
        assert sb.template_ref is None
        assert sb.total_duration_sec == 12.0
        assert sb.shots == []

    def test_with_shots(self) -> None:
        shots = [self._make_shot(i, float(i * 3), float(i * 3 + 3)) for i in range(4)]
        sb = Storyboard(
            project_id="proj-002",
            accommodation_name="Grand Hotel",
            shots=shots,
            total_duration_sec=12.0,
        )
        assert len(sb.shots) == 4

    def test_serialization_roundtrip(self) -> None:
        shots = [self._make_shot(i, float(i * 3), float(i * 3 + 3)) for i in range(2)]
        vf = VerifiedFeature(
            tag="노천탕",
            tag_en="outdoor_bath",
            confidence=0.9,
            claim_level=ClaimLevel.CONFIRMED,
        )
        sb = Storyboard(
            project_id="proj-rt",
            accommodation_name="Test Hotel",
            target_audience=TargetAudience.FAMILY,
            features=[vf],
            template_ref="tmpl-001",
            total_duration_sec=6.0,
            shots=shots,
        )
        dumped = sb.model_dump_json()
        restored = Storyboard.model_validate_json(dumped)
        assert restored.project_id == sb.project_id
        assert restored.accommodation_name == sb.accommodation_name
        assert restored.target_audience == TargetAudience.FAMILY
        assert len(restored.features) == 1
        assert restored.features[0].tag == "노천탕"
        assert restored.features[0].claim_level == ClaimLevel.CONFIRMED
        assert len(restored.shots) == 2


# ─── RenderSpec ──────────────────────────────────────────────────

class TestRenderSpec:
    def _make_storyboard(self) -> Storyboard:
        return Storyboard(project_id="proj-rs")

    def test_template_format_defaults(self) -> None:
        rs = RenderSpec(
            project_id="proj-rs",
            storyboard=self._make_storyboard(),
        )
        assert rs.format.aspect == "9:16"
        assert rs.format.fps == 30
        assert rs.format.width == 1080
        assert rs.format.height == 1920

    def test_with_music(self) -> None:
        music = MusicSpec(source="lofi.mp3", bpm_target=90.0, volume=0.4)
        rs = RenderSpec(
            project_id="proj-rs",
            storyboard=self._make_storyboard(),
            music=music,
        )
        assert rs.music is not None
        assert rs.music.bpm_target == 90.0

    def test_no_music_default(self) -> None:
        rs = RenderSpec(
            project_id="proj-rs",
            storyboard=self._make_storyboard(),
        )
        assert rs.music is None
        assert rs.captions_srt == ""
        assert rs.vo_script is None
        assert rs.output_path == ""

    def test_with_assets(self) -> None:
        asset = AssetMapping(shot_id=0, source_path="/tmp/img.jpg")
        rs = RenderSpec(
            project_id="proj-rs",
            storyboard=self._make_storyboard(),
            assets=[asset],
        )
        assert len(rs.assets) == 1
        assert rs.assets[0].transform.resize_to == (1080, 1920)
        assert rs.assets[0].transform.ken_burns is False


# ─── ProductionResult ────────────────────────────────────────────

class TestProductionResult:
    def test_complete_defaults(self) -> None:
        pr = ProductionResult(project_id="proj-001")
        assert pr.status == "complete"
        assert pr.storyboard is None
        assert pr.render_spec is None
        assert pr.features == []
        assert pr.errors == []
        assert pr.output_dir == ""

    def test_partial_status(self) -> None:
        pr = ProductionResult(
            project_id="proj-partial",
            status="partial",
            errors=["VLM timeout on image 3"],
        )
        assert pr.status == "partial"
        assert len(pr.errors) == 1

    def test_failed_status(self) -> None:
        pr = ProductionResult(
            project_id="proj-fail",
            status="failed",
            errors=["No images provided", "Template not found"],
        )
        assert pr.status == "failed"
        assert len(pr.errors) == 2

    def test_with_storyboard(self) -> None:
        sb = Storyboard(project_id="proj-001")
        pr = ProductionResult(
            project_id="proj-001",
            storyboard=sb,
            output_dir="/tmp/output/proj-001",
        )
        assert pr.storyboard is not None
        assert pr.storyboard.project_id == "proj-001"
        assert pr.output_dir == "/tmp/output/proj-001"
