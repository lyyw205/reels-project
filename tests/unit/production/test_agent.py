"""Tests for reels.production.agent.ProductionAgent."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reels.production.agent import ProductionAgent
from reels.production.models import (
    AccommodationInput,
    ClaimLevel,
    FeatureCategory,
    MatchResult,
    ProductionResult,
    RenderSpec,
    Storyboard,
    ShotCopy,
    StoryboardShot,
    ShotRole,
    TargetAudience,
    VerifiedFeature,
)
from reels.models.template import CameraType, EditInfo, TemplateFormat


# ─── Helpers ─────────────────────────────────────────────────────


def _make_verified_feature(
    tag: str = "노천탕",
    tag_en: str = "outdoor_bath",
    confidence: float = 0.85,
    claim_level: ClaimLevel = ClaimLevel.CONFIRMED,
) -> VerifiedFeature:
    return VerifiedFeature(
        tag=tag,
        tag_en=tag_en,
        confidence=confidence,
        claim_level=claim_level,
        category=FeatureCategory.AMENITY,
    )


def _make_input(name: str | None = "테스트 호텔", images: list[Path] | None = None) -> AccommodationInput:
    return AccommodationInput(
        name=name,
        images=images or [Path("img1.jpg"), Path("img2.jpg")],
        target_audience=TargetAudience.COUPLE,
    )


def _make_storyboard(project_id: str = "shorts_abc12345") -> Storyboard:
    shot = StoryboardShot(
        shot_id=0,
        role=ShotRole.HOOK,
        start_sec=0.0,
        end_sec=1.2,
        duration_sec=1.2,
        copy=ShotCopy(hook_line="특별한 숙소"),
        edit=EditInfo(),
    )
    return Storyboard(
        project_id=project_id,
        shots=[shot],
        total_duration_sec=1.2,
    )


def _make_render_spec(storyboard: Storyboard) -> RenderSpec:
    return RenderSpec(
        project_id=storyboard.project_id,
        format=TemplateFormat(),
        storyboard=storyboard,
        captions_srt="1\n00:00:00,000 --> 00:00:01,200\n특별한 숙소\n",
        vo_script="환영합니다.",
    )


def _make_agent_with_mocks(
    features: list[VerifiedFeature] | None = None,
    web_verify_enabled: bool = True,
    match_result: MatchResult | None = None,
) -> tuple[ProductionAgent, dict]:
    """Build a ProductionAgent with all sub-components mocked."""
    if features is None:
        features = [_make_verified_feature()]

    agent = ProductionAgent.__new__(ProductionAgent)

    storyboard = _make_storyboard()
    render_spec = _make_render_spec(storyboard)
    copies = [ShotCopy(hook_line="특별한 숙소")]

    mocks = {}

    # FeatureExtractor
    fe = MagicMock()
    fe.extract = AsyncMock(return_value=features)
    agent.feature_extractor = fe
    mocks["feature_extractor"] = fe

    # ClaimGate
    cg = MagicMock()
    cg.evaluate = MagicMock(return_value=features)
    cg.re_evaluate = MagicMock(return_value=features)
    agent.claim_gate = cg
    mocks["claim_gate"] = cg

    # WebVerifier
    wv = MagicMock()
    wv.enabled = web_verify_enabled
    wv.verify = AsyncMock(return_value=features)
    agent.web_verifier = wv
    mocks["web_verifier"] = wv

    # TemplateMatcher
    tm = MagicMock()
    tm.find_best = MagicMock(return_value=match_result)
    if match_result:
        archiver = MagicMock()
        archiver.get_template = MagicMock(return_value=None)
        tm.archiver = archiver
    else:
        tm.archiver = None
    agent.template_matcher = tm
    mocks["template_matcher"] = tm

    # CopyWriter (sync)
    cw = MagicMock()
    cw.generate = MagicMock(return_value=copies)
    agent.copy_writer = cw
    mocks["copy_writer"] = cw

    # StoryboardBuilder
    sb = MagicMock()
    sb.build = MagicMock(return_value=storyboard)
    agent.storyboard_builder = sb
    mocks["storyboard_builder"] = sb

    # RenderSpecGenerator
    rsg = MagicMock()
    rsg.generate = MagicMock(return_value=render_spec)
    agent.render_spec_gen = rsg
    mocks["render_spec_gen"] = rsg

    return agent, mocks


# ─── Test Cases ──────────────────────────────────────────────────


def test_produce_returns_complete_on_success():
    """produce() returns ProductionResult with status='complete' on success."""
    features = [_make_verified_feature()]
    agent, mocks = _make_agent_with_mocks(features=features)
    input_data = _make_input()

    result = asyncio.run(agent.produce(input_data))

    assert isinstance(result, ProductionResult)
    assert result.status == "complete"
    assert result.storyboard is not None
    assert result.render_spec is not None
    assert result.features == features
    assert result.errors == []


def test_produce_returns_failed_when_no_features():
    """produce() returns failed status when feature extractor returns empty list."""
    agent, mocks = _make_agent_with_mocks(features=[])
    # Override: extract returns empty
    mocks["feature_extractor"].extract = AsyncMock(return_value=[])
    input_data = _make_input()

    result = asyncio.run(agent.produce(input_data))

    assert result.status == "failed"
    assert "No features extracted from images" in result.errors[0]
    assert result.storyboard is None


def test_produce_skips_web_verification_when_no_name():
    """produce() skips web verification when input_data.name is None."""
    features = [_make_verified_feature()]
    agent, mocks = _make_agent_with_mocks(features=features, web_verify_enabled=True)
    input_data = _make_input(name=None)

    asyncio.run(agent.produce(input_data))

    mocks["web_verifier"].verify.assert_not_called()
    mocks["claim_gate"].re_evaluate.assert_not_called()


def test_produce_skips_web_verification_when_disabled():
    """produce() skips web verification when web_verifier.enabled is False."""
    features = [_make_verified_feature()]
    agent, mocks = _make_agent_with_mocks(features=features, web_verify_enabled=False)
    input_data = _make_input(name="호텔")

    asyncio.run(agent.produce(input_data))

    mocks["web_verifier"].verify.assert_not_called()
    mocks["claim_gate"].re_evaluate.assert_not_called()


def test_produce_calls_re_evaluate_after_web_verification():
    """produce() calls claim_gate.re_evaluate after web_verifier.verify."""
    features = [_make_verified_feature()]
    agent, mocks = _make_agent_with_mocks(features=features, web_verify_enabled=True)
    input_data = _make_input(name="호텔")

    asyncio.run(agent.produce(input_data))

    mocks["web_verifier"].verify.assert_called_once()
    mocks["claim_gate"].re_evaluate.assert_called_once()


def test_produce_uses_default_shot_count_when_no_template_match():
    """produce() uses shot_count=7 (default) when template_matcher returns None."""
    features = [_make_verified_feature()]
    agent, mocks = _make_agent_with_mocks(features=features, match_result=None)
    input_data = _make_input()

    asyncio.run(agent.produce(input_data))

    # copy_writer.generate should be called with shot_count=7
    args, kwargs = mocks["copy_writer"].generate.call_args
    # shot_count is the third positional arg
    assert args[2] == 7


def test_produce_saves_outputs_to_directory(tmp_path):
    """produce() creates output files when output_dir is provided."""
    features = [_make_verified_feature()]
    agent, mocks = _make_agent_with_mocks(features=features)
    input_data = _make_input()
    output_dir = tmp_path / "out"

    result = asyncio.run(agent.produce(input_data, output_dir=output_dir))

    assert result.status == "complete"
    assert (output_dir / "features.json").exists()
    assert (output_dir / "storyboard.json").exists()
    assert (output_dir / "render_spec.json").exists()
    # captions.srt and vo.txt only written when non-empty
    assert (output_dir / "captions.srt").exists()
    assert (output_dir / "vo.txt").exists()


def test_produce_handles_exception_gracefully():
    """produce() catches exceptions and returns failed ProductionResult."""
    agent, mocks = _make_agent_with_mocks()
    mocks["feature_extractor"].extract = AsyncMock(side_effect=RuntimeError("API down"))
    input_data = _make_input()

    result = asyncio.run(agent.produce(input_data))

    assert result.status == "failed"
    assert "API down" in result.errors[0]
    assert result.storyboard is None


def test_produce_works_without_template_matcher():
    """produce() works correctly when repo=None (no template_matcher)."""
    features = [_make_verified_feature()]
    agent, mocks = _make_agent_with_mocks(features=features)
    agent.template_matcher = None  # Simulate repo=None case
    input_data = _make_input()

    result = asyncio.run(agent.produce(input_data))

    assert result.status == "complete"
    # copy_writer.generate still called with default shot_count=7
    args, kwargs = mocks["copy_writer"].generate.call_args
    assert args[2] == 7


def test_save_outputs_creates_correct_files(tmp_path):
    """_save_outputs() writes features.json, storyboard.json, render_spec.json, captions.srt, vo.txt."""
    agent = ProductionAgent.__new__(ProductionAgent)
    agent.feature_extractor = MagicMock()
    agent.claim_gate = MagicMock()
    agent.copy_writer = MagicMock()
    agent.template_matcher = None
    agent.storyboard_builder = MagicMock()
    agent.web_verifier = MagicMock()
    agent.render_spec_gen = None

    storyboard = _make_storyboard()
    render_spec = _make_render_spec(storyboard)
    features = [_make_verified_feature()]

    output_dir = tmp_path / "save_test"
    agent._save_outputs(output_dir, storyboard, render_spec, features)

    # features.json is valid JSON list
    features_data = json.loads((output_dir / "features.json").read_text())
    assert isinstance(features_data, list)
    assert len(features_data) == 1
    assert features_data[0]["tag_en"] == "outdoor_bath"

    # storyboard.json is valid JSON
    sb_data = json.loads((output_dir / "storyboard.json").read_text())
    assert sb_data["project_id"] == storyboard.project_id

    # render_spec.json is valid JSON
    rs_data = json.loads((output_dir / "render_spec.json").read_text())
    assert rs_data["project_id"] == render_spec.project_id

    # captions.srt written (non-empty captions_srt)
    assert (output_dir / "captions.srt").exists()
    assert "특별한 숙소" in (output_dir / "captions.srt").read_text()

    # vo.txt written (non-empty vo_script)
    assert (output_dir / "vo.txt").exists()
    assert "환영합니다" in (output_dir / "vo.txt").read_text()
