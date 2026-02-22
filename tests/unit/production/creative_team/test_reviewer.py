"""Tests for reels.production.creative_team.reviewer.QAReviewer."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from reels.models.template import CameraType, EditInfo
from reels.production.creative_team.models import QAIssue, QAReport
from reels.production.creative_team.reviewer import QAReviewer
from reels.production.models import (
    CaptionLine,
    ClaimLevel,
    FeatureCategory,
    ShotCopy,
    ShotRole,
    Storyboard,
    StoryboardShot,
    TargetAudience,
    VerifiedFeature,
)


# ─── Helpers ─────────────────────────────────────────────────────────


def _make_verified_feature(
    tag_en: str = "outdoor_bath",
    claim_level: str = "confirmed",
    confidence: float = 0.9,
) -> VerifiedFeature:
    return VerifiedFeature(
        tag="노천탕",
        tag_en=tag_en,
        confidence=confidence,
        category=FeatureCategory.AMENITY,
        claim_level=claim_level,
    )


def _make_shot(
    shot_id: int = 0,
    role: ShotRole = ShotRole.HOOK,
    duration_sec: float = 3.0,
    hook_line: str | None = None,
    caption_lines: list[CaptionLine] | None = None,
    feature_tag: str | None = None,
) -> StoryboardShot:
    return StoryboardShot(
        shot_id=shot_id,
        role=role,
        start_sec=0.0,
        end_sec=duration_sec,
        duration_sec=duration_sec,
        feature_tag=feature_tag,
        copy=ShotCopy(
            hook_line=hook_line,
            caption_lines=caption_lines or [],
        ),
        edit=EditInfo(),
    )


def _make_storyboard(shots: list[StoryboardShot], project_id: str = "test_proj") -> Storyboard:
    total = sum(s.duration_sec for s in shots)
    return Storyboard(
        project_id=project_id,
        target_audience=TargetAudience.COUPLE,
        shots=shots,
        total_duration_sec=total,
    )


def _make_reviewer(qa_report: QAReport | None = None) -> tuple[QAReviewer, MagicMock]:
    llm = MagicMock()
    # Default: LLM returns PASS with no issues
    default_report = qa_report or QAReport(verdict="PASS")
    llm.generate = AsyncMock(return_value=default_report)
    reviewer = QAReviewer(llm=llm)
    return reviewer, llm


def _valid_storyboard() -> tuple[Storyboard, list[VerifiedFeature]]:
    """Storyboard that passes all code checks: 12s total, hook first, cta last."""
    shots = [
        _make_shot(0, ShotRole.HOOK, 3.0, hook_line="훅라인"),
        _make_shot(1, ShotRole.FEATURE, 6.0, feature_tag="outdoor_bath"),
        _make_shot(2, ShotRole.CTA, 3.0),
    ]
    features = [_make_verified_feature("outdoor_bath", "confirmed")]
    return _make_storyboard(shots), features


# ─── _run_code_checks — factual words ────────────────────────────────


def test_code_checks_factual_word_in_probable_copy():
    """PROBABLE feature + caption with '무료' → critical issue."""
    reviewer, _ = _make_reviewer()
    shots = [
        _make_shot(0, ShotRole.HOOK, 3.0, hook_line="훅"),
        _make_shot(
            1,
            ShotRole.FEATURE,
            6.0,
            caption_lines=[CaptionLine(text="무료 조식 포함")],
            feature_tag="breakfast",
        ),
        _make_shot(2, ShotRole.CTA, 3.0),
    ]
    features = [_make_verified_feature("breakfast", "probable")]
    storyboard = _make_storyboard(shots)

    issues = reviewer._run_code_checks(storyboard, features)

    critical = [i for i in issues if i.severity == "critical"]
    assert len(critical) >= 1
    # Should flag the factual word "무료"
    factual_issues = [
        i for i in critical if i.related_rule == "factual_words_confirmed_only"
    ]
    assert len(factual_issues) >= 1


def test_code_checks_confirmed_feature_with_factual_word_no_issue():
    """CONFIRMED feature + caption with '제공' → no factual word issue."""
    reviewer, _ = _make_reviewer()
    shots = [
        _make_shot(0, ShotRole.HOOK, 3.0, hook_line="훅"),
        _make_shot(
            1,
            ShotRole.FEATURE,
            6.0,
            caption_lines=[CaptionLine(text="조식 제공")],
            feature_tag="breakfast",
        ),
        _make_shot(2, ShotRole.CTA, 3.0),
    ]
    features = [_make_verified_feature("breakfast", "confirmed")]
    storyboard = _make_storyboard(shots)

    issues = reviewer._run_code_checks(storyboard, features)

    factual_issues = [
        i for i in issues if i.related_rule == "factual_words_confirmed_only"
    ]
    assert len(factual_issues) == 0


def test_code_checks_factual_word_in_hook_line():
    """PROBABLE feature + hook_line with '무제한' → critical issue."""
    reviewer, _ = _make_reviewer()
    shots = [
        _make_shot(
            0,
            ShotRole.HOOK,
            3.0,
            hook_line="무제한 힐링",
            feature_tag="spa",
        ),
        _make_shot(1, ShotRole.FEATURE, 6.0),
        _make_shot(2, ShotRole.CTA, 3.0),
    ]
    features = [_make_verified_feature("spa", "probable")]
    storyboard = _make_storyboard(shots)

    issues = reviewer._run_code_checks(storyboard, features)
    factual_issues = [
        i for i in issues if i.related_rule == "factual_words_confirmed_only"
    ]
    assert len(factual_issues) >= 1


# ─── _run_code_checks — hook_line char limit ─────────────────────────


def test_code_checks_hook_line_too_long():
    """hook_line > 12 chars → warning issue."""
    reviewer, _ = _make_reviewer()
    # 13 Korean characters — exceeds the 12-char limit
    shots = [
        _make_shot(0, ShotRole.HOOK, 3.0, hook_line="이것은너무긴훅라인입니다요"),
        _make_shot(1, ShotRole.FEATURE, 6.0),
        _make_shot(2, ShotRole.CTA, 3.0),
    ]
    storyboard = _make_storyboard(shots)

    issues = reviewer._run_code_checks(storyboard, [])

    hook_issues = [i for i in issues if i.related_rule == "hook_max_chars"]
    assert len(hook_issues) == 1
    assert hook_issues[0].severity == "warning"


def test_code_checks_hook_line_exact_12_no_issue():
    """hook_line of exactly 12 chars → no warning."""
    reviewer, _ = _make_reviewer()
    # Exactly 12 Korean characters
    shots = [
        _make_shot(0, ShotRole.HOOK, 3.0, hook_line="일이삼사오육칠팔구십십일십이"),
        _make_shot(1, ShotRole.FEATURE, 6.0),
        _make_shot(2, ShotRole.CTA, 3.0),
    ]
    storyboard = _make_storyboard(shots)
    # "일이삼사오육칠팔구십십일십이" is 14 chars, use a 12-char one
    shots[0] = _make_shot(0, ShotRole.HOOK, 3.0, hook_line="일이삼사오육칠팔구십십일")
    storyboard = _make_storyboard(shots)

    issues = reviewer._run_code_checks(storyboard, [])
    hook_issues = [i for i in issues if i.related_rule == "hook_max_chars"]
    assert len(hook_issues) == 0


# ─── _run_code_checks — caption char limit ────────────────────────────


def test_code_checks_caption_too_long():
    """caption > 14 chars → warning issue."""
    reviewer, _ = _make_reviewer()
    long_caption = "이것은정말너무긴캡션라인입니다"  # > 14 chars
    shots = [
        _make_shot(0, ShotRole.HOOK, 3.0, hook_line="훅"),
        _make_shot(
            1, ShotRole.FEATURE, 6.0,
            caption_lines=[CaptionLine(text=long_caption)],
        ),
        _make_shot(2, ShotRole.CTA, 3.0),
    ]
    storyboard = _make_storyboard(shots)

    issues = reviewer._run_code_checks(storyboard, [])
    caption_issues = [i for i in issues if i.related_rule == "caption_max_chars"]
    assert len(caption_issues) == 1
    assert caption_issues[0].severity == "warning"


def test_code_checks_caption_ok_no_issue():
    """caption <= 14 chars → no issue."""
    reviewer, _ = _make_reviewer()
    shots = [
        _make_shot(0, ShotRole.HOOK, 3.0, hook_line="훅"),
        _make_shot(
            1, ShotRole.FEATURE, 6.0,
            caption_lines=[CaptionLine(text="짧은캡션")],
        ),
        _make_shot(2, ShotRole.CTA, 3.0),
    ]
    storyboard = _make_storyboard(shots)

    issues = reviewer._run_code_checks(storyboard, [])
    caption_issues = [i for i in issues if i.related_rule == "caption_max_chars"]
    assert len(caption_issues) == 0


# ─── _run_code_checks — total duration ───────────────────────────────


def test_code_checks_duration_too_short():
    """Total duration < 10s → critical issue."""
    reviewer, _ = _make_reviewer()
    shots = [
        _make_shot(0, ShotRole.HOOK, 2.0),
        _make_shot(1, ShotRole.FEATURE, 3.0),
        _make_shot(2, ShotRole.CTA, 2.0),
    ]  # total = 7s
    storyboard = _make_storyboard(shots)

    issues = reviewer._run_code_checks(storyboard, [])
    dur_issues = [i for i in issues if i.related_rule == "duration_range"]
    assert len(dur_issues) == 1
    assert dur_issues[0].severity == "critical"


def test_code_checks_duration_too_long():
    """Total duration > 15s → critical issue."""
    reviewer, _ = _make_reviewer()
    shots = [
        _make_shot(0, ShotRole.HOOK, 6.0),
        _make_shot(1, ShotRole.FEATURE, 6.0),
        _make_shot(2, ShotRole.CTA, 6.0),
    ]  # total = 18s
    storyboard = _make_storyboard(shots)

    issues = reviewer._run_code_checks(storyboard, [])
    dur_issues = [i for i in issues if i.related_rule == "duration_range"]
    assert len(dur_issues) == 1
    assert dur_issues[0].severity == "critical"


def test_code_checks_duration_ok_no_issue():
    """Total duration 10-15s → no duration issue."""
    reviewer, _ = _make_reviewer()
    shots = [
        _make_shot(0, ShotRole.HOOK, 4.0),
        _make_shot(1, ShotRole.FEATURE, 5.0),
        _make_shot(2, ShotRole.CTA, 3.0),
    ]  # total = 12s
    storyboard = _make_storyboard(shots)

    issues = reviewer._run_code_checks(storyboard, [])
    dur_issues = [i for i in issues if i.related_rule == "duration_range"]
    assert len(dur_issues) == 0


# ─── _run_code_checks — structure ─────────────────────────────────────


def test_code_checks_first_shot_not_hook():
    """First shot not hook → critical issue."""
    reviewer, _ = _make_reviewer()
    shots = [
        _make_shot(0, ShotRole.FEATURE, 4.0),
        _make_shot(1, ShotRole.FEATURE, 5.0),
        _make_shot(2, ShotRole.CTA, 3.0),
    ]
    storyboard = _make_storyboard(shots)

    issues = reviewer._run_code_checks(storyboard, [])
    struct_issues = [i for i in issues if i.related_rule == "structure_first_hook"]
    assert len(struct_issues) == 1
    assert struct_issues[0].severity == "critical"


def test_code_checks_last_shot_not_cta():
    """Last shot not cta → critical issue."""
    reviewer, _ = _make_reviewer()
    shots = [
        _make_shot(0, ShotRole.HOOK, 3.0),
        _make_shot(1, ShotRole.FEATURE, 5.0),
        _make_shot(2, ShotRole.FEATURE, 4.0),
    ]
    storyboard = _make_storyboard(shots)

    issues = reviewer._run_code_checks(storyboard, [])
    struct_issues = [i for i in issues if i.related_rule == "structure_last_cta"]
    assert len(struct_issues) == 1
    assert struct_issues[0].severity == "critical"


def test_code_checks_valid_storyboard_has_no_critical_issues():
    """Valid storyboard (12s, hook first, cta last, no violations) → no critical."""
    reviewer, _ = _make_reviewer()
    storyboard, features = _valid_storyboard()

    issues = reviewer._run_code_checks(storyboard, features)
    critical = [i for i in issues if i.severity == "critical"]
    assert len(critical) == 0


# ─── review() — async, with LLM ──────────────────────────────────────


def test_review_returns_pass_when_no_critical_issues():
    """review() returns PASS when code checks produce no critical issues and LLM says PASS."""
    llm_report = QAReport(verdict="PASS", issues=[])
    reviewer, _ = _make_reviewer(llm_report)
    storyboard, features = _valid_storyboard()

    async def _run():
        return await reviewer.review(storyboard, features, revision_count=0)

    result = asyncio.run(_run())
    assert isinstance(result, QAReport)
    assert result.verdict == "PASS"


def test_review_returns_revise_when_code_critical_issues_present():
    """review() returns REVISE when code checks find critical issues."""
    llm_report = QAReport(verdict="PASS", issues=[])
    reviewer, _ = _make_reviewer(llm_report)

    # Storyboard with total duration < 10s (critical issue)
    shots = [
        _make_shot(0, ShotRole.HOOK, 2.0),
        _make_shot(1, ShotRole.FEATURE, 3.0),
        _make_shot(2, ShotRole.CTA, 2.0),
    ]
    features = []
    storyboard = _make_storyboard(shots)

    async def _run():
        return await reviewer.review(storyboard, features)

    result = asyncio.run(_run())
    assert result.verdict == "REVISE"


def test_review_returns_pass_with_only_warnings():
    """review() returns PASS when only warnings exist (no criticals)."""
    # LLM returns PASS with a warning
    warning_issue = QAIssue(
        severity="warning",
        responsible_agent="writer",
        target_element="shots[0].copy.hook_line",
        current_value="이것은너무긴훅라인",
        expected_behavior="hook_line 최대 12자",
    )
    llm_report = QAReport(verdict="PASS", issues=[warning_issue])
    reviewer, _ = _make_reviewer(llm_report)

    # Valid storyboard — code checks produce no criticals, only warnings from LLM
    storyboard, features = _valid_storyboard()
    # Add a long hook line for warnings
    storyboard.shots[0] = _make_shot(0, ShotRole.HOOK, 3.0, hook_line="이것은너무긴훅라인")
    # Total duration still valid

    async def _run():
        return await reviewer.review(storyboard, features, revision_count=0)

    result = asyncio.run(_run())
    # Only warnings → PASS
    assert result.verdict == "PASS"


def test_review_qa_issue_has_responsible_agent_field():
    """QAIssues in QAReport have a responsible_agent field."""
    code_issue = QAIssue(
        severity="critical",
        responsible_agent="director",
        target_element="total_duration_sec",
        current_value="8.0",
        expected_behavior="10-15초",
        related_rule="duration_range",
    )
    assert code_issue.responsible_agent == "director"

    llm_issue = QAIssue(
        severity="warning",
        responsible_agent="writer",
        target_element="shots[0].copy.hook_line",
        current_value="짧은 훅",
        expected_behavior="더 강렬한 훅",
    )
    assert llm_issue.responsible_agent == "writer"


def test_review_merges_code_and_llm_issues():
    """review() merges code_issues and llm_report.issues together."""
    # LLM returns 1 issue
    llm_issue = QAIssue(
        severity="warning",
        responsible_agent="writer",
        target_element="narrative_flow",
        current_value="weak",
        expected_behavior="stronger narrative",
    )
    llm_report = QAReport(verdict="PASS", issues=[llm_issue])
    reviewer, _ = _make_reviewer(llm_report)

    # Storyboard that generates 1 code warning (hook too long)
    shots = [
        _make_shot(0, ShotRole.HOOK, 4.0, hook_line="이것은너무긴훅라인입니다요"),
        _make_shot(1, ShotRole.FEATURE, 4.0),
        _make_shot(2, ShotRole.CTA, 4.0),
    ]
    storyboard = _make_storyboard(shots)

    async def _run():
        return await reviewer.review(storyboard, [], revision_count=0)

    result = asyncio.run(_run())
    # Should have both code issue and LLM issue
    assert len(result.issues) >= 2


def test_review_sets_revision_count():
    """review() stores revision_count in the returned QAReport."""
    llm_report = QAReport(verdict="PASS")
    reviewer, _ = _make_reviewer(llm_report)
    storyboard, features = _valid_storyboard()

    async def _run():
        return await reviewer.review(storyboard, features, revision_count=2)

    result = asyncio.run(_run())
    assert result.revision_count == 2
