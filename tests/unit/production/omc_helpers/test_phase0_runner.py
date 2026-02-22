"""Tests for reels.production.omc_helpers.phase0_runner."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reels.production.models import (
    AccommodationInput,
    ClaimLevel,
    Feature,
    FeatureCategory,
    TargetAudience,
    VerifiedFeature,
)
from reels.production.omc_helpers.phase0_runner import (
    _build_input,
    run_phase0,
    save_phase0_outputs,
)


# ── _build_input tests ───────────────────────────────────────────────


class TestBuildInput:
    def test_basic_input(self):
        result = _build_input(
            images=["img1.jpg", "img2.jpg"],
            name="테스트숙소",
            region="제주",
            target="couple",
        )
        assert isinstance(result, AccommodationInput)
        assert result.name == "테스트숙소"
        assert result.region == "제주"
        assert result.target_audience == TargetAudience.COUPLE
        assert len(result.images) == 2
        assert result.images[0] == Path("img1.jpg")

    def test_default_target(self):
        result = _build_input(images=["img.jpg"])
        assert result.target_audience == TargetAudience.COUPLE
        assert result.name is None
        assert result.region is None

    def test_family_target(self):
        result = _build_input(images=["img.jpg"], target="family")
        assert result.target_audience == TargetAudience.FAMILY


# ── save_phase0_outputs tests ────────────────────────────────────────


class TestSaveOutputs:
    def test_saves_features_and_context(self, tmp_path: Path):
        features = [
            VerifiedFeature(
                tag="노천탕",
                tag_en="outdoor_bath",
                confidence=0.85,
                claim_level=ClaimLevel.CONFIRMED,
                category=FeatureCategory.AMENITY,
                copy_tone="단정",
            ),
        ]
        input_data = AccommodationInput(
            name="테스트숙소",
            region="제주",
            target_audience=TargetAudience.COUPLE,
            images=[Path("img1.jpg")],
        )

        feat_path, ctx_path = save_phase0_outputs(tmp_path, features, input_data)

        assert feat_path.exists()
        assert ctx_path.exists()

        feat_data = json.loads(feat_path.read_text())
        assert isinstance(feat_data, list)
        assert len(feat_data) == 1
        assert feat_data[0]["tag_en"] == "outdoor_bath"

        ctx_data = json.loads(ctx_path.read_text())
        assert ctx_data["name"] == "테스트숙소"
        assert ctx_data["target_audience"] == "couple"

    def test_creates_directory_if_missing(self, tmp_path: Path):
        output_dir = tmp_path / "nested" / "dir"
        features: list[VerifiedFeature] = []
        input_data = AccommodationInput(images=[Path("img.jpg")])

        feat_path, ctx_path = save_phase0_outputs(output_dir, features, input_data)
        assert output_dir.exists()
        assert feat_path.exists()


# ── run_phase0 tests ─────────────────────────────────────────────────


class TestRunPhase0:
    def test_successful_extraction(self):
        async def _run():
            mock_features = [
                Feature(
                    tag="수영장",
                    tag_en="pool",
                    confidence=0.9,
                    category=FeatureCategory.AMENITY,
                ),
            ]
            mock_verified = [
                VerifiedFeature(
                    tag="수영장",
                    tag_en="pool",
                    confidence=0.9,
                    claim_level=ClaimLevel.CONFIRMED,
                    category=FeatureCategory.AMENITY,
                ),
            ]

            with (
                patch("reels.production.feature_extractor.FeatureExtractor") as MockFE,
                patch("reels.production.claim_gate.ClaimGate") as MockCG,
                patch("reels.production.web_verifier.WebVerifier") as MockWV,
            ):
                mock_ext = MockFE.return_value
                mock_ext.extract = AsyncMock(return_value=mock_features)
                MockCG.return_value.evaluate.return_value = mock_verified
                MockWV.return_value.enabled = False

                input_data = AccommodationInput(
                    name="테스트",
                    images=[Path("img.jpg")],
                )
                verified, ctx = await run_phase0(input_data, no_web=True)

                assert len(verified) == 1
                assert verified[0].tag_en == "pool"
                mock_ext.extract.assert_called_once()

        asyncio.run(_run())

    def test_no_features_raises(self):
        async def _run():
            with (
                patch("reels.production.feature_extractor.FeatureExtractor") as MockFE,
                patch("reels.production.claim_gate.ClaimGate"),
                patch("reels.production.web_verifier.WebVerifier"),
            ):
                mock_ext = MockFE.return_value
                mock_ext.extract = AsyncMock(return_value=[])

                input_data = AccommodationInput(images=[Path("img.jpg")])

                with pytest.raises(RuntimeError, match="No features"):
                    await run_phase0(input_data)

        asyncio.run(_run())

    def test_web_verification_called_when_enabled(self):
        async def _run():
            mock_features = [
                Feature(tag="뷰", tag_en="view", confidence=0.6, category=FeatureCategory.VIEW),
            ]
            mock_verified = [
                VerifiedFeature(tag="뷰", tag_en="view", confidence=0.6, claim_level=ClaimLevel.PROBABLE, category=FeatureCategory.VIEW),
            ]

            with (
                patch("reels.production.feature_extractor.FeatureExtractor") as MockFE,
                patch("reels.production.claim_gate.ClaimGate") as MockCG,
                patch("reels.production.web_verifier.WebVerifier") as MockWV,
            ):
                mock_ext = MockFE.return_value
                mock_ext.extract = AsyncMock(return_value=mock_features)
                mock_cg = MockCG.return_value
                mock_cg.evaluate.return_value = mock_verified
                mock_cg.re_evaluate.return_value = mock_verified
                mock_wv = MockWV.return_value
                mock_wv.enabled = True
                mock_wv.verify = AsyncMock(return_value=mock_verified)

                input_data = AccommodationInput(
                    name="테스트숙소",
                    images=[Path("img.jpg")],
                )
                verified, _ = await run_phase0(input_data, no_web=False)

                mock_wv.verify.assert_called_once()
                mock_cg.re_evaluate.assert_called_once()

        asyncio.run(_run())

    def test_web_verification_skipped_when_no_name(self):
        async def _run():
            mock_features = [
                Feature(tag="뷰", tag_en="view", confidence=0.6, category=FeatureCategory.VIEW),
            ]
            mock_verified = [
                VerifiedFeature(tag="뷰", tag_en="view", confidence=0.6, claim_level=ClaimLevel.PROBABLE, category=FeatureCategory.VIEW),
            ]

            with (
                patch("reels.production.feature_extractor.FeatureExtractor") as MockFE,
                patch("reels.production.claim_gate.ClaimGate") as MockCG,
                patch("reels.production.web_verifier.WebVerifier") as MockWV,
            ):
                mock_ext = MockFE.return_value
                mock_ext.extract = AsyncMock(return_value=mock_features)
                MockCG.return_value.evaluate.return_value = mock_verified
                mock_wv = MockWV.return_value
                mock_wv.enabled = True
                mock_wv.verify = AsyncMock()

                input_data = AccommodationInput(
                    name=None,  # no name
                    images=[Path("img.jpg")],
                )
                await run_phase0(input_data)

                mock_wv.verify.assert_not_called()

        asyncio.run(_run())
