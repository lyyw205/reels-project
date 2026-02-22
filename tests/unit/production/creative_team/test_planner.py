"""Tests for reels.production.creative_team.planner.CreativePlanner."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from reels.production.creative_team.models import CreativeBrief, NarrativeStructure, QAIssue
from reels.production.creative_team.planner import CreativePlanner
from reels.production.models import (
    AccommodationInput,
    ClaimLevel,
    FeatureCategory,
    TargetAudience,
    VerifiedFeature,
)


# ─── Helpers ─────────────────────────────────────────────────────────


def _make_verified_feature(tag_en="outdoor_bath", claim_level="confirmed", confidence=0.9):
    return VerifiedFeature(
        tag="노천탕",
        tag_en=tag_en,
        confidence=confidence,
        category=FeatureCategory.AMENITY,
        claim_level=claim_level,
    )


def _make_brief():
    return CreativeBrief(
        concept_keywords=["비밀스러운", "프라이빗"],
        hero_features=["outdoor_bath", "ocean_view"],
        narrative_structure=NarrativeStructure.REVEAL,
        target_emotion="궁금증",
        tone_direction="고급",
        hook_direction="이런곳이?",
        cta_direction="예약",
    )


def _make_context():
    return AccommodationInput(
        name="테스트 리조트",
        region="제주",
        target_audience=TargetAudience.COUPLE,
        images=[Path("img1.jpg"), Path("img2.jpg")],
    )


def _make_planner(brief_to_return: CreativeBrief) -> tuple[CreativePlanner, MagicMock]:
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=brief_to_return)
    planner = CreativePlanner(llm=llm)
    return planner, llm


# ─── plan() ──────────────────────────────────────────────────────────


def test_plan_returns_creative_brief():
    """plan() returns the CreativeBrief produced by the LLM."""
    brief = _make_brief()
    planner, _ = _make_planner(brief)
    features = [_make_verified_feature()]
    context = _make_context()

    async def _run():
        return await planner.plan(features, context)

    result = asyncio.run(_run())
    assert isinstance(result, CreativeBrief)
    assert result.narrative_structure == NarrativeStructure.REVEAL
    assert "outdoor_bath" in result.hero_features


def test_plan_calls_llm_generate_with_planner_role():
    """plan() passes agent_role='planner' to LLM.generate."""
    brief = _make_brief()
    planner, llm = _make_planner(brief)
    features = [_make_verified_feature()]
    context = _make_context()

    async def _run():
        return await planner.plan(features, context)

    asyncio.run(_run())

    _, kwargs = llm.generate.call_args
    assert kwargs.get("agent_role") == "planner"


def test_plan_passes_response_model_creative_brief():
    """plan() requests CreativeBrief as response_model."""
    brief = _make_brief()
    planner, llm = _make_planner(brief)
    features = [_make_verified_feature()]
    context = _make_context()

    async def _run():
        return await planner.plan(features, context)

    asyncio.run(_run())

    _, kwargs = llm.generate.call_args
    assert kwargs.get("response_model") is CreativeBrief


def test_plan_includes_features_in_prompt():
    """plan() builds a prompt that mentions feature tag_en values."""
    brief = _make_brief()
    planner, llm = _make_planner(brief)
    features = [_make_verified_feature("ocean_view")]
    context = _make_context()

    async def _run():
        return await planner.plan(features, context)

    asyncio.run(_run())

    _, kwargs = llm.generate.call_args
    assert "ocean_view" in kwargs.get("prompt", "")


def test_plan_with_match_result_includes_template_info():
    """plan() includes template info in prompt when match_result is provided."""
    brief = _make_brief()
    planner, llm = _make_planner(brief)
    features = [_make_verified_feature()]
    context = _make_context()

    match_result = MagicMock()
    match_result.template_id = "tmpl_abc"
    match_result.shot_count = 6
    match_result.template_duration = 12.0

    async def _run():
        return await planner.plan(features, context, match_result)

    asyncio.run(_run())

    _, kwargs = llm.generate.call_args
    assert "tmpl_abc" in kwargs.get("prompt", "")


# ─── revise() ────────────────────────────────────────────────────────


def test_revise_returns_creative_brief():
    """revise() returns a CreativeBrief from the LLM."""
    brief = _make_brief()
    planner, _ = _make_planner(brief)
    issue = QAIssue(
        severity="critical",
        responsible_agent="planner",
        target_element="narrative_structure",
        current_value="reveal",
        expected_behavior="highlight",
    )

    async def _run():
        return await planner.revise(brief, [issue])

    result = asyncio.run(_run())
    assert isinstance(result, CreativeBrief)


def test_revise_passes_previous_brief_in_prompt():
    """revise() includes previous brief JSON in the prompt."""
    brief = _make_brief()
    planner, llm = _make_planner(brief)
    issue = QAIssue(
        severity="warning",
        responsible_agent="planner",
        target_element="concept_keywords",
        current_value="['비밀스러운']",
        expected_behavior="더 강렬한 키워드",
        suggestion="'럭셔리'로 교체",
    )

    async def _run():
        return await planner.revise(brief, [issue])

    asyncio.run(_run())

    _, kwargs = llm.generate.call_args
    prompt = kwargs.get("prompt", "")
    # Previous brief JSON should be in the prompt
    assert "concept_keywords" in prompt


def test_revise_calls_llm_with_planner_role():
    """revise() also passes agent_role='planner' to LLM.generate."""
    brief = _make_brief()
    planner, llm = _make_planner(brief)
    issue = QAIssue(
        severity="warning",
        responsible_agent="planner",
        target_element="tone_direction",
        current_value="고급",
        expected_behavior="친근한",
    )

    async def _run():
        return await planner.revise(brief, [issue])

    asyncio.run(_run())

    _, kwargs = llm.generate.call_args
    assert kwargs.get("agent_role") == "planner"


def test_revise_includes_issue_details_in_prompt():
    """revise() includes issue expected_behavior in the prompt."""
    brief = _make_brief()
    planner, llm = _make_planner(brief)
    issue = QAIssue(
        severity="critical",
        responsible_agent="planner",
        target_element="hook_direction",
        current_value="이런곳이?",
        expected_behavior="더 강렬한 훅 필요",
    )

    async def _run():
        return await planner.revise(brief, [issue])

    asyncio.run(_run())

    _, kwargs = llm.generate.call_args
    assert "더 강렬한 훅 필요" in kwargs.get("prompt", "")
