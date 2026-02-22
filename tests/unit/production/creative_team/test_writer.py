"""Tests for reels.production.creative_team.writer.CreativeWriter."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from reels.production.creative_team.models import (
    CreativeBrief,
    NarrativeStructure,
    PlannedShot,
    QAIssue,
    ShotPlan,
)
from reels.production.creative_team.writer import CreativeWriter
from reels.production.models import (
    AccommodationInput,
    CaptionLine,
    ClaimLevel,
    FeatureCategory,
    ShotCopy,
    ShotRole,
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


def _make_shot_plan(n_shots: int = 3) -> ShotPlan:
    roles = [ShotRole.HOOK, ShotRole.FEATURE, ShotRole.CTA]
    shots = [
        PlannedShot(
            shot_id=i,
            role=roles[i % len(roles)],
            duration_sec=2.0,
            feature_tag="outdoor_bath" if i == 1 else None,
            image_index=i % 2,
        )
        for i in range(n_shots)
    ]
    return ShotPlan(shots=shots, total_duration_sec=float(n_shots * 2))


def _make_context():
    return AccommodationInput(
        name="테스트 리조트",
        region="강원",
        target_audience=TargetAudience.COUPLE,
        images=[Path("img0.jpg"), Path("img1.jpg")],
    )


def _make_copies(n: int) -> list[ShotCopy]:
    copies = []
    for i in range(n):
        if i == 0:
            copies.append(ShotCopy(hook_line=f"훅{i}"))
        else:
            copies.append(ShotCopy(
                caption_lines=[CaptionLine(text=f"캡션{i}")]
            ))
    return copies


class _CopiesResponse:
    """Minimal mock for _CopiesResponse Pydantic model."""
    def __init__(self, copies):
        self.copies = copies


def _make_writer(copies_to_return: list[ShotCopy]) -> tuple[CreativeWriter, MagicMock]:
    llm = MagicMock()
    # _CopiesResponse wrapper
    response = _CopiesResponse(copies_to_return)
    llm.generate = AsyncMock(return_value=response)
    writer = CreativeWriter(llm=llm)
    return writer, llm


# ─── write() ─────────────────────────────────────────────────────────


def test_write_returns_list_of_shot_copy():
    """write() returns a list with one ShotCopy per shot."""
    copies = _make_copies(3)
    writer, _ = _make_writer(copies)
    brief = _make_brief()
    shot_plan = _make_shot_plan(3)
    features = [_make_verified_feature()]
    context = _make_context()

    async def _run():
        return await writer.write(brief, shot_plan, features, context)

    result = asyncio.run(_run())
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(c, ShotCopy) for c in result)


def test_write_calls_llm_with_writer_role():
    """write() passes agent_role='writer' to LLM.generate."""
    copies = _make_copies(3)
    writer, llm = _make_writer(copies)
    brief = _make_brief()
    shot_plan = _make_shot_plan(3)
    features = [_make_verified_feature()]
    context = _make_context()

    async def _run():
        return await writer.write(brief, shot_plan, features, context)

    asyncio.run(_run())
    _, kwargs = llm.generate.call_args
    assert kwargs.get("agent_role") == "writer"


def test_write_passes_use_cache_false():
    """write() always passes use_cache=False to LLM."""
    copies = _make_copies(3)
    writer, llm = _make_writer(copies)
    brief = _make_brief()
    shot_plan = _make_shot_plan(3)
    features = [_make_verified_feature()]
    context = _make_context()

    async def _run():
        return await writer.write(brief, shot_plan, features, context)

    asyncio.run(_run())
    _, kwargs = llm.generate.call_args
    assert kwargs.get("use_cache") is False


def test_write_pads_when_llm_returns_fewer_copies():
    """If LLM returns fewer copies than shots, pad with empty ShotCopy."""
    # 2 copies for 3 shots
    copies = _make_copies(2)
    writer, _ = _make_writer(copies)
    brief = _make_brief()
    shot_plan = _make_shot_plan(3)
    features = [_make_verified_feature()]
    context = _make_context()

    async def _run():
        return await writer.write(brief, shot_plan, features, context)

    result = asyncio.run(_run())
    assert len(result) == 3
    # The padded copy should be an empty ShotCopy
    assert result[2].hook_line is None
    assert result[2].caption_lines == []


def test_write_truncates_when_llm_returns_more_copies():
    """If LLM returns more copies than shots, truncate."""
    # 5 copies for 3 shots
    copies = _make_copies(5)
    writer, _ = _make_writer(copies)
    brief = _make_brief()
    shot_plan = _make_shot_plan(3)
    features = [_make_verified_feature()]
    context = _make_context()

    async def _run():
        return await writer.write(brief, shot_plan, features, context)

    result = asyncio.run(_run())
    assert len(result) == 3


# ─── revise() ────────────────────────────────────────────────────────


def test_revise_filters_writer_issues_only():
    """revise() only sends writer-owned issues to LLM; others are ignored."""
    copies = _make_copies(3)
    writer, llm = _make_writer(copies)
    shot_plan = _make_shot_plan(3)

    issues = [
        QAIssue(
            severity="critical",
            responsible_agent="director",  # NOT writer
            target_element="shots[0].role",
            current_value="feature",
            expected_behavior="hook",
        ),
        QAIssue(
            severity="warning",
            responsible_agent="writer",
            target_element="shots[1].copy.hook_line",
            current_value="무료 제공",
            expected_behavior="CONFIRMED만 사용",
        ),
    ]

    async def _run():
        return await writer.revise(copies, shot_plan, issues)

    result = asyncio.run(_run())
    # LLM was called (writer issue present)
    assert llm.generate.called
    # Result should be aligned to shot count
    assert len(result) == 3


def test_revise_skips_llm_when_no_writer_issues():
    """revise() returns previous copies directly when no writer issues exist."""
    copies = _make_copies(3)
    writer, llm = _make_writer(copies)
    shot_plan = _make_shot_plan(3)

    issues = [
        QAIssue(
            severity="critical",
            responsible_agent="director",
            target_element="shots[0].role",
            current_value="feature",
            expected_behavior="hook",
        ),
    ]

    async def _run():
        return await writer.revise(copies, shot_plan, issues)

    result = asyncio.run(_run())
    assert not llm.generate.called
    assert len(result) == 3


def test_revise_includes_writer_issue_details_in_prompt():
    """revise() includes expected_behavior of writer issues in prompt."""
    copies = _make_copies(2)
    writer, llm = _make_writer(copies)
    shot_plan = _make_shot_plan(2)

    issues = [
        QAIssue(
            severity="critical",
            responsible_agent="writer",
            target_element="shots[0].copy.hook_line",
            current_value="무료 제공",
            expected_behavior="CONFIRMED 레벨에서만 무료 사용가능",
        ),
    ]

    async def _run():
        return await writer.revise(copies, shot_plan, issues)

    asyncio.run(_run())
    _, kwargs = llm.generate.call_args
    assert "CONFIRMED 레벨에서만 무료 사용가능" in kwargs.get("prompt", "")


def test_revise_aligns_result_length():
    """revise() result is always aligned to shot_plan length."""
    # LLM returns 1 copy, plan has 3 shots
    copies_from_llm = [ShotCopy(hook_line="훅")]
    writer, _ = _make_writer(copies_from_llm)
    previous = _make_copies(3)
    shot_plan = _make_shot_plan(3)

    issues = [
        QAIssue(
            severity="warning",
            responsible_agent="writer",
            target_element="shots[0].copy.hook_line",
            current_value="긴 텍스트",
            expected_behavior="12자 이내",
        ),
    ]

    async def _run():
        return await writer.revise(previous, shot_plan, issues)

    result = asyncio.run(_run())
    assert len(result) == 3
