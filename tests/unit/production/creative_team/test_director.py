"""Tests for reels.production.creative_team.director.ProducerDirector."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from reels.production.creative_team.director import ProducerDirector, _clamp_image_indices
from reels.production.creative_team.models import (
    CreativeBrief,
    NarrativeStructure,
    PlannedShot,
    QAIssue,
    ShotPlan,
)
from reels.production.models import (
    ClaimLevel,
    FeatureCategory,
    ShotRole,
    TargetAudience,
    VerifiedFeature,
)


# ─── Helpers ─────────────────────────────────────────────────────────


def _make_verified_feature(tag_en="outdoor_bath", confidence=0.9):
    return VerifiedFeature(
        tag="노천탕",
        tag_en=tag_en,
        confidence=confidence,
        category=FeatureCategory.AMENITY,
        claim_level=ClaimLevel.CONFIRMED,
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


def _make_shot_plan(image_indices: list[int] | None = None):
    indices = image_indices or [0, 1, 0]
    shots = [
        PlannedShot(
            shot_id=i,
            role=[ShotRole.HOOK, ShotRole.FEATURE, ShotRole.CTA][i],
            duration_sec=2.0,
            image_index=indices[i],
        )
        for i in range(3)
    ]
    return ShotPlan(shots=shots, total_duration_sec=6.0)


def _make_director(shot_plan: ShotPlan) -> tuple[ProducerDirector, MagicMock]:
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=shot_plan)
    director = ProducerDirector(llm=llm)
    return director, llm


# ─── _clamp_image_indices ────────────────────────────────────────────


def test_clamp_image_indices_within_range():
    plan = _make_shot_plan([0, 1, 0])
    result = _clamp_image_indices(plan, max_index=2)
    assert [s.image_index for s in result.shots] == [0, 1, 0]


def test_clamp_image_indices_above_max():
    plan = _make_shot_plan([0, 5, 10])
    result = _clamp_image_indices(plan, max_index=2)
    assert all(s.image_index <= 2 for s in result.shots)
    assert result.shots[1].image_index == 2
    assert result.shots[2].image_index == 2


def test_clamp_image_indices_negative():
    plan = _make_shot_plan([-1, 0, 1])
    result = _clamp_image_indices(plan, max_index=1)
    assert result.shots[0].image_index == 0


def test_clamp_recomputes_total_duration():
    plan = _make_shot_plan([0, 1, 0])
    result = _clamp_image_indices(plan, max_index=1)
    expected_total = sum(s.duration_sec for s in result.shots)
    assert result.total_duration_sec == expected_total


# ─── direct() ────────────────────────────────────────────────────────


def test_direct_returns_shot_plan():
    """direct() returns a ShotPlan from the LLM."""
    plan = _make_shot_plan()
    director, _ = _make_director(plan)
    brief = _make_brief()
    features = [_make_verified_feature()]
    assets = [Path("img0.jpg"), Path("img1.jpg")]

    async def _run():
        return await director.direct(brief, features, assets)

    result = asyncio.run(_run())
    assert isinstance(result, ShotPlan)
    assert len(result.shots) == 3


def test_direct_calls_llm_with_director_role():
    """direct() passes agent_role='director' to LLM."""
    plan = _make_shot_plan()
    director, llm = _make_director(plan)
    brief = _make_brief()
    features = [_make_verified_feature()]
    assets = [Path("img0.jpg")]

    async def _run():
        return await director.direct(brief, features, assets)

    asyncio.run(_run())

    _, kwargs = llm.generate.call_args
    assert kwargs.get("agent_role") == "director"


def test_direct_clamps_image_indices_to_asset_count():
    """direct() clamps image_index values to [0, len(assets)-1]."""
    # LLM returns a plan with out-of-range image indices
    shots = [
        PlannedShot(shot_id=0, role=ShotRole.HOOK, duration_sec=2.0, image_index=0),
        PlannedShot(shot_id=1, role=ShotRole.FEATURE, duration_sec=2.0, image_index=99),
        PlannedShot(shot_id=2, role=ShotRole.CTA, duration_sec=2.0, image_index=50),
    ]
    plan = ShotPlan(shots=shots, total_duration_sec=6.0)
    director, _ = _make_director(plan)
    brief = _make_brief()
    features = [_make_verified_feature()]
    # Only 2 assets → max_index = 1
    assets = [Path("img0.jpg"), Path("img1.jpg")]

    async def _run():
        return await director.direct(brief, features, assets)

    result = asyncio.run(_run())
    assert all(s.image_index <= 1 for s in result.shots)
    assert result.shots[1].image_index == 1
    assert result.shots[2].image_index == 1


def test_direct_single_asset_clamps_all_to_zero():
    """With only 1 asset, all image_index must be 0."""
    shots = [
        PlannedShot(shot_id=0, role=ShotRole.HOOK, duration_sec=2.0, image_index=3),
        PlannedShot(shot_id=1, role=ShotRole.CTA, duration_sec=2.0, image_index=7),
    ]
    plan = ShotPlan(shots=shots, total_duration_sec=4.0)
    director, _ = _make_director(plan)
    brief = _make_brief()
    features = []
    assets = [Path("img0.jpg")]

    async def _run():
        return await director.direct(brief, features, assets)

    result = asyncio.run(_run())
    assert all(s.image_index == 0 for s in result.shots)


def test_direct_with_template():
    """direct() includes template info in prompt when template provided."""
    plan = _make_shot_plan()
    director, llm = _make_director(plan)
    brief = _make_brief()
    features = [_make_verified_feature()]
    assets = [Path("img0.jpg")]

    template = MagicMock()
    template.template_id = "tmpl_xyz"
    template.shot_count = 6
    template.total_duration_sec = 12.0

    async def _run():
        return await director.direct(brief, features, assets, template)

    asyncio.run(_run())

    _, kwargs = llm.generate.call_args
    assert "tmpl_xyz" in kwargs.get("prompt", "")


# ─── revise() ────────────────────────────────────────────────────────


def test_revise_returns_shot_plan():
    """revise() returns a revised ShotPlan."""
    plan = _make_shot_plan()
    director, _ = _make_director(plan)
    issue = QAIssue(
        severity="critical",
        responsible_agent="director",
        target_element="shots[0].role",
        current_value="feature",
        expected_behavior="hook",
    )

    async def _run():
        return await director.revise(plan, [issue])

    result = asyncio.run(_run())
    assert isinstance(result, ShotPlan)


def test_revise_calls_llm_with_use_cache_false():
    """revise() passes use_cache=False to LLM (always fresh generation)."""
    plan = _make_shot_plan()
    director, llm = _make_director(plan)
    issue = QAIssue(
        severity="warning",
        responsible_agent="director",
        target_element="total_duration_sec",
        current_value="8.0",
        expected_behavior="10-15초",
    )

    async def _run():
        return await director.revise(plan, [issue])

    asyncio.run(_run())

    _, kwargs = llm.generate.call_args
    assert kwargs.get("use_cache") is False


def test_revise_includes_issues_in_prompt():
    """revise() includes issue details in the prompt."""
    plan = _make_shot_plan()
    director, llm = _make_director(plan)
    issue = QAIssue(
        severity="critical",
        responsible_agent="director",
        target_element="shots[2].role",
        current_value="feature",
        expected_behavior="마지막 샷은 cta",
    )

    async def _run():
        return await director.revise(plan, [issue])

    asyncio.run(_run())

    _, kwargs = llm.generate.call_args
    assert "마지막 샷은 cta" in kwargs.get("prompt", "")


def test_revise_clamps_image_indices():
    """revise() applies clamping based on max image_index in previous plan."""
    previous = _make_shot_plan([0, 1, 0])  # max existing = 1
    # LLM returns out-of-range indices
    shots = [
        PlannedShot(shot_id=0, role=ShotRole.HOOK, duration_sec=2.0, image_index=0),
        PlannedShot(shot_id=1, role=ShotRole.FEATURE, duration_sec=2.0, image_index=5),
        PlannedShot(shot_id=2, role=ShotRole.CTA, duration_sec=2.0, image_index=3),
    ]
    revised_plan = ShotPlan(shots=shots, total_duration_sec=6.0)

    director, llm = _make_director(revised_plan)
    issue = QAIssue(
        severity="warning",
        responsible_agent="director",
        target_element="pacing",
        current_value="slow",
        expected_behavior="faster pacing",
    )

    async def _run():
        return await director.revise(previous, [issue])

    result = asyncio.run(_run())
    # max existing in previous = 1
    assert all(s.image_index <= 1 for s in result.shots)
