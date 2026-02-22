"""Integration tests for reels.production.creative_team.team_agent.CreativeTeamAgent."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from reels.models.template import EditInfo
from reels.production.creative_team.models import (
    CreativeBrief,
    NarrativeStructure,
    PlannedShot,
    QAIssue,
    QAReport,
    ShotPlan,
)
from reels.production.creative_team.team_agent import CreativeTeamAgent
from reels.production.models import (
    AccommodationInput,
    ClaimLevel,
    FeatureCategory,
    ProductionResult,
    ShotCopy,
    ShotRole,
    Storyboard,
    StoryboardShot,
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


def _make_shot_plan():
    shots = [
        PlannedShot(shot_id=0, role=ShotRole.HOOK, duration_sec=3.0, image_index=0),
        PlannedShot(shot_id=1, role=ShotRole.FEATURE, duration_sec=6.0, feature_tag="outdoor_bath", image_index=1),
        PlannedShot(shot_id=2, role=ShotRole.CTA, duration_sec=3.0, image_index=0),
    ]
    return ShotPlan(shots=shots, total_duration_sec=12.0)


def _make_copies():
    return [
        ShotCopy(hook_line="이런곳이?"),
        ShotCopy(),
        ShotCopy(),
    ]


def _make_storyboard(project_id: str = "test_proj") -> Storyboard:
    shots = [
        StoryboardShot(
            shot_id=0,
            role=ShotRole.HOOK,
            start_sec=0.0,
            end_sec=3.0,
            duration_sec=3.0,
            copy=ShotCopy(hook_line="이런곳이?"),
            edit=EditInfo(),
        ),
        StoryboardShot(
            shot_id=1,
            role=ShotRole.FEATURE,
            start_sec=3.0,
            end_sec=9.0,
            duration_sec=6.0,
            feature_tag="outdoor_bath",
            copy=ShotCopy(),
            edit=EditInfo(),
        ),
        StoryboardShot(
            shot_id=2,
            role=ShotRole.CTA,
            start_sec=9.0,
            end_sec=12.0,
            duration_sec=3.0,
            copy=ShotCopy(),
            edit=EditInfo(),
        ),
    ]
    return Storyboard(
        project_id=project_id,
        target_audience=TargetAudience.COUPLE,
        shots=shots,
        total_duration_sec=12.0,
    )


def _make_agent_with_mocks(
    features: list[VerifiedFeature] | None = None,
    web_verify_enabled: bool = False,
    reviewer_reports: list[QAReport] | None = None,
    max_revisions: int = 2,
) -> tuple[CreativeTeamAgent, dict]:
    """Build a CreativeTeamAgent with all sub-components mocked."""
    if features is None:
        features = [_make_verified_feature()]

    # Use __new__ to bypass __init__ (which would try to create real LLM clients)
    agent = CreativeTeamAgent.__new__(CreativeTeamAgent)
    agent._max_revisions = max_revisions

    mocks: dict = {}

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

    # Template matcher — disabled
    agent.template_matcher = None

    # CopyWriter (for factual claim checking in _post_validate_copies)
    cw = MagicMock()
    cw.check_factual_claims = MagicMock(return_value=[])
    cw.sanitize_factual = MagicMock(side_effect=lambda text, _: text)
    agent.copy_writer = cw
    mocks["copy_writer"] = cw

    # CreativeLLM (not used directly — agents are mocked)
    agent._llm = MagicMock()

    # Planner
    planner = MagicMock()
    planner.plan = AsyncMock(return_value=_make_brief())
    planner.revise = AsyncMock(return_value=_make_brief())
    agent._planner = planner
    mocks["planner"] = planner

    # Director
    director = MagicMock()
    director.direct = AsyncMock(return_value=_make_shot_plan())
    director.revise = AsyncMock(return_value=_make_shot_plan())
    agent._director = director
    mocks["director"] = director

    # Writer
    writer = MagicMock()
    writer.write = AsyncMock(return_value=_make_copies())
    writer.revise = AsyncMock(return_value=_make_copies())
    agent._writer = writer
    mocks["writer"] = writer

    # Reviewer — configure sequence of QAReports
    if reviewer_reports is None:
        reviewer_reports = [QAReport(verdict="PASS")]

    reviewer = MagicMock()
    reviewer.review = AsyncMock(side_effect=reviewer_reports)
    agent._reviewer = reviewer
    mocks["reviewer"] = reviewer

    # RenderSpecGenerator — disabled for simplicity
    agent.render_spec_gen = None

    return agent, mocks


def _make_input(
    name: str | None = "테스트 리조트",
    images: list[Path] | None = None,
) -> AccommodationInput:
    return AccommodationInput(
        name=name,
        images=images or [Path("img0.jpg"), Path("img1.jpg")],
        target_audience=TargetAudience.COUPLE,
        region="제주",
    )


# ─── Full pipeline: produce() → status=complete ───────────────────────


def test_produce_returns_complete_on_success():
    """produce() returns ProductionResult with status='complete' on success."""
    agent, _ = _make_agent_with_mocks()
    input_data = _make_input()

    async def _run():
        return await agent.produce(input_data)

    result = asyncio.run(_run())
    assert isinstance(result, ProductionResult)
    assert result.status == "complete"
    assert result.storyboard is not None
    assert result.errors == []


def test_produce_storyboard_has_correct_shot_count():
    """produce() storyboard contains shots matching the shot plan."""
    agent, _ = _make_agent_with_mocks()
    input_data = _make_input()

    async def _run():
        return await agent.produce(input_data)

    result = asyncio.run(_run())
    assert result.storyboard is not None
    assert len(result.storyboard.shots) == 3  # matches _make_shot_plan()


def test_produce_calls_each_agent_in_order():
    """produce() invokes planner, then director, then writer, then reviewer."""
    call_order: list[str] = []

    agent, mocks = _make_agent_with_mocks()
    input_data = _make_input()

    async def planner_plan(*args, **kwargs):
        call_order.append("planner")
        return _make_brief()

    async def director_direct(*args, **kwargs):
        call_order.append("director")
        return _make_shot_plan()

    async def writer_write(*args, **kwargs):
        call_order.append("writer")
        return _make_copies()

    async def reviewer_review(*args, **kwargs):
        call_order.append("reviewer")
        return QAReport(verdict="PASS")

    mocks["planner"].plan = planner_plan
    mocks["director"].direct = director_direct
    mocks["writer"].write = writer_write
    mocks["reviewer"].review = reviewer_review

    async def _run():
        return await agent.produce(input_data)

    asyncio.run(_run())

    assert call_order.index("planner") < call_order.index("director")
    assert call_order.index("director") < call_order.index("writer")
    assert call_order.index("writer") < call_order.index("reviewer")


# ─── Failure: no features → status=failed ────────────────────────────


def test_produce_returns_failed_when_no_features():
    """produce() returns failed status when feature extractor returns empty list."""
    agent, mocks = _make_agent_with_mocks(features=[])
    mocks["feature_extractor"].extract = AsyncMock(return_value=[])
    input_data = _make_input()

    async def _run():
        return await agent.produce(input_data)

    result = asyncio.run(_run())
    assert result.status == "failed"
    assert any("No features" in e for e in result.errors)
    assert result.storyboard is None


def test_produce_returns_failed_on_exception():
    """produce() catches unexpected exceptions and returns status='failed'."""
    agent, mocks = _make_agent_with_mocks()
    mocks["feature_extractor"].extract = AsyncMock(
        side_effect=RuntimeError("API unavailable")
    )
    input_data = _make_input()

    async def _run():
        return await agent.produce(input_data)

    result = asyncio.run(_run())
    assert result.status == "failed"
    assert "API unavailable" in result.errors[0]


# ─── REVISE feedback loop ─────────────────────────────────────────────


def test_produce_revise_then_pass():
    """Reviewer returns REVISE on first call, PASS on second → one revision round."""
    revise_report = QAReport(
        verdict="REVISE",
        issues=[
            QAIssue(
                severity="critical",
                responsible_agent="director",
                target_element="total_duration_sec",
                current_value="8.0",
                expected_behavior="10-15초",
                related_rule="duration_range",
            )
        ],
    )
    pass_report = QAReport(verdict="PASS")

    agent, mocks = _make_agent_with_mocks(
        reviewer_reports=[revise_report, pass_report],
        max_revisions=2,
    )
    input_data = _make_input()

    async def _run():
        return await agent.produce(input_data)

    result = asyncio.run(_run())
    assert result.status == "complete"
    # Reviewer was called twice (first REVISE, then PASS)
    assert mocks["reviewer"].review.call_count == 2


def test_produce_revise_routes_to_director():
    """REVISE with director issue → director.revise is called."""
    revise_report = QAReport(
        verdict="REVISE",
        issues=[
            QAIssue(
                severity="critical",
                responsible_agent="director",
                target_element="total_duration_sec",
                current_value="8.0",
                expected_behavior="10-15초",
                related_rule="duration_range",
            )
        ],
    )
    pass_report = QAReport(verdict="PASS")

    agent, mocks = _make_agent_with_mocks(
        reviewer_reports=[revise_report, pass_report],
        max_revisions=2,
    )
    input_data = _make_input()

    async def _run():
        return await agent.produce(input_data)

    asyncio.run(_run())
    # Director revise should have been called
    mocks["director"].revise.assert_called_once()


def test_produce_revise_routes_to_writer():
    """REVISE with writer-only issue → writer.revise is called (not director)."""
    revise_report = QAReport(
        verdict="REVISE",
        issues=[
            QAIssue(
                severity="critical",
                responsible_agent="writer",
                target_element="shots[0].copy.hook_line",
                current_value="무료 제공",
                expected_behavior="CONFIRMED 레벨에서만 무료 사용",
                related_rule="factual_words_confirmed_only",
            )
        ],
    )
    pass_report = QAReport(verdict="PASS")

    agent, mocks = _make_agent_with_mocks(
        reviewer_reports=[revise_report, pass_report],
        max_revisions=2,
    )
    input_data = _make_input()

    async def _run():
        return await agent.produce(input_data)

    asyncio.run(_run())
    # Writer revise should be called; director revise should NOT
    mocks["writer"].revise.assert_called_once()
    mocks["director"].revise.assert_not_called()


# ─── Max revision cap ─────────────────────────────────────────────────


def test_produce_max_revisions_forces_pass():
    """Reviewer always returns REVISE → forced PASS after max_revisions."""
    always_revise = QAReport(
        verdict="REVISE",
        issues=[
            QAIssue(
                severity="critical",
                responsible_agent="director",
                target_element="total_duration_sec",
                current_value="8.0",
                expected_behavior="10-15초",
                related_rule="duration_range",
            )
        ],
    )
    # 3 REVISE reports to saturate max_revisions=2
    agent, mocks = _make_agent_with_mocks(
        reviewer_reports=[always_revise, always_revise, always_revise],
        max_revisions=2,
    )
    input_data = _make_input()

    async def _run():
        return await agent.produce(input_data)

    result = asyncio.run(_run())
    # Should still complete (forced PASS)
    assert result.status == "complete"
    # Reviewer called max_revisions+1 times (initial + revisions)
    assert mocks["reviewer"].review.call_count == 3


def test_produce_max_revisions_one_revision():
    """With max_revisions=1, REVISE loop stops after 1 revision."""
    always_revise = QAReport(
        verdict="REVISE",
        issues=[
            QAIssue(
                severity="critical",
                responsible_agent="director",
                target_element="shots[0].role",
                current_value="feature",
                expected_behavior="hook",
            )
        ],
    )
    agent, mocks = _make_agent_with_mocks(
        reviewer_reports=[always_revise, always_revise],
        max_revisions=1,
    )
    input_data = _make_input()

    async def _run():
        return await agent.produce(input_data)

    result = asyncio.run(_run())
    assert result.status == "complete"
    assert mocks["reviewer"].review.call_count == 2  # initial + 1 revision


# ─── Output directory saving ──────────────────────────────────────────


def test_produce_saves_storyboard_to_output_dir(tmp_path):
    """produce() writes storyboard.json when output_dir is provided."""
    agent, _ = _make_agent_with_mocks()
    input_data = _make_input()
    output_dir = tmp_path / "out"

    async def _run():
        return await agent.produce(input_data, output_dir=output_dir)

    result = asyncio.run(_run())
    assert result.status == "complete"
    assert (output_dir / "storyboard.json").exists()
    assert (output_dir / "features.json").exists()


# ─── Planner cascade revision ─────────────────────────────────────────


def test_produce_revise_planner_cascades_to_director_and_writer():
    """REVISE with planner issue → planner.revise + director.direct + writer.write all re-run."""
    revise_report = QAReport(
        verdict="REVISE",
        issues=[
            QAIssue(
                severity="critical",
                responsible_agent="planner",
                target_element="narrative_structure",
                current_value="reveal",
                expected_behavior="highlight",
            )
        ],
    )
    pass_report = QAReport(verdict="PASS")

    agent, mocks = _make_agent_with_mocks(
        reviewer_reports=[revise_report, pass_report],
        max_revisions=2,
    )
    input_data = _make_input()

    async def _run():
        return await agent.produce(input_data)

    asyncio.run(_run())

    # planner.revise must be called
    mocks["planner"].revise.assert_called_once()
    # director.direct is called at least twice (initial + cascade from planner revision)
    assert mocks["director"].direct.call_count >= 2
    # writer.write is called at least twice too
    assert mocks["writer"].write.call_count >= 2
