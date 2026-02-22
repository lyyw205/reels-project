"""Tests for reels.production.storyboard_builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from reels.models.template import (
    AudioInfo,
    CameraInfo,
    CameraType,
    EditInfo,
    Template,
    TemplateShot,
    TransitionType,
)
from reels.production.models import (
    AccommodationInput,
    ClaimLevel,
    ShotCopy,
    ShotRole,
    VerifiedFeature,
)
from reels.production.storyboard_builder import StoryboardBuilder
from reels.production.template_matcher import DEFAULT_STRUCTURE


# ─── Helpers ─────────────────────────────────────────────────────


def _make_feature(
    tag_en: str = "outdoor_bath",
    confidence: float = 0.8,
    evidence_images: list[str] | None = None,
    claim_level: ClaimLevel = ClaimLevel.CONFIRMED,
) -> VerifiedFeature:
    return VerifiedFeature(
        tag=tag_en,
        tag_en=tag_en,
        confidence=confidence,
        evidence_images=evidence_images or [],
        claim_level=claim_level,
    )


def _make_template(
    template_id: str = "t1",
    duration: float = 12.0,
    shot_count: int = 7,
    camera_types: list[CameraType] | None = None,
) -> Template:
    cam_list = camera_types or [CameraType.STATIC]
    shot_dur = duration / shot_count
    shots = []
    for i in range(shot_count):
        shots.append(
            TemplateShot(
                shot_id=i,
                start_sec=i * shot_dur,
                end_sec=(i + 1) * shot_dur,
                duration_sec=shot_dur,
                place_label="other",
                camera=CameraInfo(type=cam_list[i % len(cam_list)]),
                audio=AudioInfo(has_speech=False),
                keyframe_paths=[],
                bpm=90.0,
            )
        )
    return Template(
        template_id=template_id,
        total_duration_sec=duration,
        shot_count=shot_count,
        shots=shots,
    )


def _make_context(name: str = "Test Hotel") -> AccommodationInput:
    return AccommodationInput(name=name, images=[Path("/tmp/a.jpg")])


def _make_builder() -> StoryboardBuilder:
    return StoryboardBuilder()


# ─── _assign_roles ───────────────────────────────────────────────


class TestAssignRoles:
    def test_assign_roles_7_shots(self) -> None:
        """7 shots, 5 features → HOOK, FEATURE x3, SUPPORT x2, CTA."""
        builder = _make_builder()
        # 5 features: feature_count=3 (capped), support_count=min(2, 5-3)=2
        features = [_make_feature(f"feat_{i}") for i in range(5)]
        roles = builder._assign_roles(features, shot_count=7)

        assert len(roles) == 7
        assert roles[0] == ShotRole.HOOK
        assert roles[1] == ShotRole.FEATURE
        assert roles[2] == ShotRole.FEATURE
        assert roles[3] == ShotRole.FEATURE
        assert roles[4] == ShotRole.SUPPORT
        assert roles[5] == ShotRole.SUPPORT
        assert roles[6] == ShotRole.CTA

    def test_assign_roles_few_features(self) -> None:
        """1 feature → HOOK, FEATURE, CTA x5."""
        builder = _make_builder()
        features = [_make_feature("pool")]
        roles = builder._assign_roles(features, shot_count=7)

        assert len(roles) == 7
        assert roles[0] == ShotRole.HOOK
        assert roles[1] == ShotRole.FEATURE
        # Shots 2-6 get CTA (feature_count=1, support_count=0)
        for i in range(2, 7):
            assert roles[i] == ShotRole.CTA, f"roles[{i}] should be CTA, got {roles[i]}"


# ─── build — timing ──────────────────────────────────────────────


class TestBuildTiming:
    def test_build_uses_template_timing(self) -> None:
        """Template with specific durations → preserved in shots."""
        builder = _make_builder()
        # 3-shot template with distinct durations
        shots = [
            TemplateShot(
                shot_id=0, start_sec=0.0, end_sec=1.0, duration_sec=1.0,
                place_label="other", camera=CameraInfo(type=CameraType.STATIC),
                audio=AudioInfo(), keyframe_paths=[],
            ),
            TemplateShot(
                shot_id=1, start_sec=1.0, end_sec=3.5, duration_sec=2.5,
                place_label="other", camera=CameraInfo(type=CameraType.PAN_LEFT),
                audio=AudioInfo(), keyframe_paths=[],
            ),
            TemplateShot(
                shot_id=2, start_sec=3.5, end_sec=5.0, duration_sec=1.5,
                place_label="other", camera=CameraInfo(type=CameraType.STATIC),
                audio=AudioInfo(), keyframe_paths=[],
            ),
        ]
        template = Template(
            template_id="timing-t", total_duration_sec=5.0, shot_count=3, shots=shots,
        )
        features = [_make_feature("lobby")]
        storyboard = builder.build(
            features=features,
            copies=[],
            template=template,
            assets=[],
            context=_make_context(),
        )

        assert len(storyboard.shots) == 3
        assert storyboard.shots[0].duration_sec == pytest.approx(1.0, abs=1e-3)
        assert storyboard.shots[1].duration_sec == pytest.approx(2.5, abs=1e-3)
        assert storyboard.shots[2].duration_sec == pytest.approx(1.5, abs=1e-3)

    def test_build_fallback_no_template(self) -> None:
        """No template → uses DEFAULT_STRUCTURE (7 shots)."""
        builder = _make_builder()
        features = [_make_feature("pool")]
        storyboard = builder.build(
            features=features,
            copies=[],
            template=None,
            assets=[],
            context=_make_context(),
        )

        assert len(storyboard.shots) == len(DEFAULT_STRUCTURE)
        assert len(storyboard.shots) == 7
        assert storyboard.template_ref is None

    def test_build_total_duration(self) -> None:
        """Sum of shot durations matches total_duration_sec."""
        builder = _make_builder()
        features = [_make_feature("view")]
        storyboard = builder.build(
            features=features,
            copies=[],
            template=None,
            assets=[],
            context=_make_context(),
        )

        computed = round(sum(s.duration_sec for s in storyboard.shots), 3)
        assert storyboard.total_duration_sec == pytest.approx(computed, abs=1e-3)


# ─── asset matching ──────────────────────────────────────────────


class TestAssetMatching:
    def test_asset_matching_by_evidence(self) -> None:
        """Feature evidence_images maps to correct asset path."""
        builder = _make_builder()
        assets = [
            Path("/tmp/img_pool.jpg"),
            Path("/tmp/img_view.jpg"),
            Path("/tmp/img_lobby.jpg"),
        ]
        features = [
            _make_feature("pool", evidence_images=["img_pool.jpg"]),
            _make_feature("view", evidence_images=["img_view.jpg"]),
            _make_feature("lobby", evidence_images=["img_lobby.jpg"]),
        ]
        roles = [ShotRole.HOOK, ShotRole.FEATURE, ShotRole.FEATURE, ShotRole.FEATURE,
                 ShotRole.SUPPORT, ShotRole.SUPPORT, ShotRole.CTA]
        asset_map = builder._match_assets_to_shots(features, assets, roles)

        # HOOK uses features[0] (pool) → img_pool.jpg
        assert asset_map.get(0) == Path("/tmp/img_pool.jpg")
        # Shot 1 (FEATURE) uses features[0] again but pool is used; next evidence
        # Shot 2 uses features[1] (view) → img_view.jpg
        assert asset_map.get(2) == Path("/tmp/img_view.jpg")

    def test_unused_assets_fill_remaining_shots(self) -> None:
        """Shots without matched evidence get filled with unused assets."""
        builder = _make_builder()
        assets = [Path("/tmp/a.jpg"), Path("/tmp/b.jpg"), Path("/tmp/c.jpg")]
        features = [_make_feature("pool")]  # No evidence_images
        roles = [ShotRole.HOOK, ShotRole.FEATURE, ShotRole.CTA]
        asset_map = builder._match_assets_to_shots(features, assets, roles)

        # All 3 assets should be distributed across shots
        assert len(asset_map) == 3


# ─── hook gets best feature ──────────────────────────────────────


class TestHookGetsBestFeature:
    def test_hook_gets_best_feature(self) -> None:
        """Shot 0 (HOOK) always gets features[0]."""
        builder = _make_builder()
        features = [
            _make_feature("outdoor_bath"),
            _make_feature("pool"),
            _make_feature("view"),
        ]
        storyboard = builder.build(
            features=features,
            copies=[],
            template=None,
            assets=[],
            context=_make_context(),
        )

        hook_shot = storyboard.shots[0]
        assert hook_shot.role == ShotRole.HOOK
        assert hook_shot.feature_tag == "outdoor_bath"


# ─── CTA has no feature ──────────────────────────────────────────


class TestCtaHasNoFeature:
    def test_cta_has_no_feature(self) -> None:
        """CTA shots have feature_tag=None."""
        builder = _make_builder()
        features = [_make_feature(f"feat_{i}") for i in range(3)]
        storyboard = builder.build(
            features=features,
            copies=[],
            template=None,
            assets=[],
            context=_make_context(),
        )

        cta_shots = [s for s in storyboard.shots if s.role == ShotRole.CTA]
        assert len(cta_shots) >= 1
        for shot in cta_shots:
            assert shot.feature_tag is None
            assert shot.claim_level is None


# ─── camera from template ────────────────────────────────────────


class TestCameraFromTemplate:
    def test_camera_from_template(self) -> None:
        """Template camera types are carried through to storyboard shots."""
        builder = _make_builder()
        camera_sequence = [
            CameraType.PUSH_IN,
            CameraType.PAN_LEFT,
            CameraType.GIMBAL_SMOOTH,
        ]
        template = _make_template(
            template_id="cam-t",
            duration=6.0,
            shot_count=3,
            camera_types=camera_sequence,
        )
        features = [_make_feature("lobby")]
        storyboard = builder.build(
            features=features,
            copies=[],
            template=template,
            assets=[],
            context=_make_context(),
        )

        assert len(storyboard.shots) == 3
        assert storyboard.shots[0].camera_suggestion == CameraType.PUSH_IN
        assert storyboard.shots[1].camera_suggestion == CameraType.PAN_LEFT
        assert storyboard.shots[2].camera_suggestion == CameraType.GIMBAL_SMOOTH


# ─── _is_video ───────────────────────────────────────────────────


class TestIsVideoDetection:
    @pytest.mark.parametrize("filename,expected", [
        ("clip.mp4", True),
        ("clip.mov", True),
        ("clip.avi", True),
        ("clip.mkv", True),
        ("clip.webm", True),
        ("photo.jpg", False),
        ("photo.jpeg", False),
        ("photo.png", False),
        ("photo.heic", False),
        ("", False),
    ])
    def test_is_video_detection(self, filename: str, expected: bool) -> None:
        """.mp4 → True, .jpg → False, etc."""
        assert StoryboardBuilder._is_video(filename) is expected

    def test_is_video_with_path_object(self) -> None:
        assert StoryboardBuilder._is_video(Path("/tmp/video.mp4")) is True
        assert StoryboardBuilder._is_video(Path("/tmp/photo.jpg")) is False
