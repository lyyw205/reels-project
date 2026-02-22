"""Tests for reels.production.creative_team.models."""

from __future__ import annotations

import pytest

from reels.models.template import CameraType, TransitionType
from reels.production.creative_team.models import (
    CreativeBrief,
    NarrativeStructure,
    PlannedShot,
    QAIssue,
    QAReport,
    ShotPlan,
)
from reels.production.models import ClaimLevel, ShotCopy, ShotRole


# ─── NarrativeStructure ──────────────────────────────────────────────


def test_narrative_structure_values():
    assert NarrativeStructure.REVEAL == "reveal"
    assert NarrativeStructure.HIGHLIGHT == "highlight"
    assert NarrativeStructure.STORY == "story"
    assert NarrativeStructure.COMPARISON == "comparison"


def test_narrative_structure_is_str():
    # StrEnum: instance is also a str
    assert isinstance(NarrativeStructure.REVEAL, str)


# ─── CreativeBrief ───────────────────────────────────────────────────


def test_creative_brief_creation():
    brief = CreativeBrief(
        concept_keywords=["비밀스러운", "프라이빗"],
        hero_features=["outdoor_bath", "ocean_view"],
        narrative_structure=NarrativeStructure.REVEAL,
        target_emotion="궁금증",
        tone_direction="고급",
        hook_direction="이런곳이?",
        cta_direction="예약",
    )
    assert brief.narrative_structure == NarrativeStructure.REVEAL
    assert "outdoor_bath" in brief.hero_features
    assert len(brief.concept_keywords) == 2


def test_creative_brief_min_length_concept_keywords():
    """concept_keywords requires at least 1 item."""
    with pytest.raises(Exception):
        CreativeBrief(
            concept_keywords=[],
            hero_features=["outdoor_bath"],
            narrative_structure=NarrativeStructure.HIGHLIGHT,
            target_emotion="설레임",
            tone_direction="따뜻한",
            hook_direction="어서오세요",
            cta_direction="예약하기",
        )


def test_creative_brief_min_length_hero_features():
    """hero_features requires at least 1 item."""
    with pytest.raises(Exception):
        CreativeBrief(
            concept_keywords=["비밀스러운"],
            hero_features=[],
            narrative_structure=NarrativeStructure.STORY,
            target_emotion="편안함",
            tone_direction="아늑한",
            hook_direction="쉼표",
            cta_direction="지금 예약",
        )


def test_creative_brief_max_length_concept_keywords():
    """concept_keywords max is 5."""
    with pytest.raises(Exception):
        CreativeBrief(
            concept_keywords=["a", "b", "c", "d", "e", "f"],
            hero_features=["outdoor_bath"],
            narrative_structure=NarrativeStructure.COMPARISON,
            target_emotion="놀람",
            tone_direction="반전",
            hook_direction="설마?",
            cta_direction="바로 예약",
        )


# ─── PlannedShot ─────────────────────────────────────────────────────


def test_planned_shot_creation():
    shot = PlannedShot(
        shot_id=0,
        role=ShotRole.HOOK,
        duration_sec=1.5,
        feature_tag="outdoor_bath",
        camera=CameraType.PUSH_IN,
        image_index=0,
    )
    assert shot.shot_id == 0
    assert shot.role == ShotRole.HOOK
    assert shot.duration_sec == 1.5
    assert shot.camera == CameraType.PUSH_IN


def test_planned_shot_defaults():
    shot = PlannedShot(shot_id=1, role=ShotRole.CTA, duration_sec=2.0)
    assert shot.camera == CameraType.STATIC
    assert shot.image_index == 0
    assert shot.feature_tag is None
    assert shot.transition == TransitionType.CUT
    assert shot.visual_direction == ""


def test_planned_shot_duration_gt_zero():
    with pytest.raises(Exception):
        PlannedShot(shot_id=0, role=ShotRole.HOOK, duration_sec=0.0)


def test_planned_shot_duration_le_ten():
    with pytest.raises(Exception):
        PlannedShot(shot_id=0, role=ShotRole.HOOK, duration_sec=10.1)


def test_planned_shot_to_storyboard_shot():
    shot = PlannedShot(
        shot_id=2,
        role=ShotRole.FEATURE,
        duration_sec=2.5,
        feature_tag="ocean_view",
        camera=CameraType.PAN_LEFT,
        image_index=1,
        transition=TransitionType.DISSOLVE,
    )
    copy = ShotCopy(hook_line="오션뷰")
    sb_shot = shot.to_storyboard_shot(
        copy=copy,
        asset_path="/img/ocean.jpg",
        claim_level=ClaimLevel.CONFIRMED,
        start_sec=3.0,
    )

    assert sb_shot.shot_id == 2
    assert sb_shot.role == ShotRole.FEATURE
    assert sb_shot.start_sec == 3.0
    assert sb_shot.end_sec == pytest.approx(5.5)
    assert sb_shot.duration_sec == 2.5
    assert sb_shot.asset_type == "image"
    assert sb_shot.asset_path == "/img/ocean.jpg"
    assert sb_shot.feature_tag == "ocean_view"
    assert sb_shot.claim_level == ClaimLevel.CONFIRMED
    assert sb_shot.place_label == "ocean_view"
    assert sb_shot.camera_suggestion == CameraType.PAN_LEFT
    assert sb_shot.shot_copy.hook_line == "오션뷰"
    assert sb_shot.edit.transition_out == TransitionType.DISSOLVE


def test_planned_shot_to_storyboard_shot_explicit_end_sec():
    """When end_sec is provided, it overrides start_sec + duration_sec."""
    shot = PlannedShot(shot_id=0, role=ShotRole.HOOK, duration_sec=1.5)
    copy = ShotCopy()
    sb_shot = shot.to_storyboard_shot(
        copy=copy, asset_path="img.jpg", start_sec=0.0, end_sec=2.0
    )
    assert sb_shot.end_sec == 2.0


def test_planned_shot_to_storyboard_shot_no_feature_tag():
    """place_label defaults to 'other' when feature_tag is None."""
    shot = PlannedShot(shot_id=0, role=ShotRole.CTA, duration_sec=1.5)
    copy = ShotCopy()
    sb_shot = shot.to_storyboard_shot(copy=copy, asset_path="img.jpg")
    assert sb_shot.place_label == "other"
    assert sb_shot.feature_tag is None


# ─── ShotPlan ────────────────────────────────────────────────────────


def test_shot_plan_creation():
    shots = [
        PlannedShot(shot_id=0, role=ShotRole.HOOK, duration_sec=1.5),
        PlannedShot(shot_id=1, role=ShotRole.FEATURE, duration_sec=2.0),
        PlannedShot(shot_id=2, role=ShotRole.CTA, duration_sec=1.5),
    ]
    plan = ShotPlan(shots=shots, total_duration_sec=5.0)
    assert len(plan.shots) == 3
    assert plan.total_duration_sec == 5.0
    assert plan.pacing_note == ""
    assert plan.music_mood == ""


def test_shot_plan_requires_at_least_one_shot():
    with pytest.raises(Exception):
        ShotPlan(shots=[], total_duration_sec=5.0)


def test_shot_plan_total_duration_gt_zero():
    shots = [PlannedShot(shot_id=0, role=ShotRole.HOOK, duration_sec=1.5)]
    with pytest.raises(Exception):
        ShotPlan(shots=shots, total_duration_sec=0.0)


# ─── QAIssue ─────────────────────────────────────────────────────────


def test_qa_issue_creation():
    issue = QAIssue(
        severity="critical",
        responsible_agent="writer",
        target_element="shots[0].copy.hook_line",
        current_value="무료 제공 무제한",
        expected_behavior="CONFIRMED 레벨에서만 '무료' 사용 가능",
    )
    assert issue.severity == "critical"
    assert issue.responsible_agent == "writer"
    assert issue.target_element == "shots[0].copy.hook_line"
    assert issue.related_rule is None
    assert issue.suggestion == ""


def test_qa_issue_all_severity_values():
    for sev in ("critical", "warning", "suggestion"):
        issue = QAIssue(
            severity=sev,
            responsible_agent="director",
            target_element="total_duration_sec",
            current_value="9.0",
            expected_behavior="10-15초",
        )
        assert issue.severity == sev


def test_qa_issue_all_responsible_agents():
    for agent in ("planner", "director", "writer"):
        issue = QAIssue(
            severity="warning",
            responsible_agent=agent,
            target_element="shots[0].role",
            current_value="feature",
            expected_behavior="hook",
        )
        assert issue.responsible_agent == agent


def test_qa_issue_invalid_responsible_agent():
    with pytest.raises(Exception):
        QAIssue(
            severity="warning",
            responsible_agent="reviewer",  # not valid
            target_element="x",
            current_value="y",
            expected_behavior="z",
        )


# ─── QAReport ────────────────────────────────────────────────────────


def test_qa_report_pass():
    report = QAReport(verdict="PASS")
    assert report.verdict == "PASS"
    assert report.issues == []
    assert report.revision_count == 0


def test_qa_report_revise():
    issue = QAIssue(
        severity="critical",
        responsible_agent="director",
        target_element="shots[0].role",
        current_value="feature",
        expected_behavior="hook",
    )
    report = QAReport(verdict="REVISE", issues=[issue], revision_count=1)
    assert report.verdict == "REVISE"
    assert len(report.issues) == 1
    assert report.revision_count == 1
