"""Pydantic v2 models for the creative team pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from reels.models.template import CameraType, EditInfo, TransitionType
from reels.production.models import ClaimLevel, ShotCopy, ShotRole


class NarrativeStructure(StrEnum):
    """Narrative structure type for shorts.

    Config-driven selection heuristics in config/default.yaml:
      creative_team.narrative_structures.<type>.preferred_when
    """

    REVEAL = "reveal"  # exterior -> interior -> hero amenity
    HIGHLIGHT = "highlight"  # best feature first -> support -> CTA
    STORY = "story"  # check-in -> room -> dining -> healing
    COMPARISON = "comparison"  # plain exterior -> stunning interior


class CreativeBrief(BaseModel):
    """Creative concept document from the Planner agent."""

    concept_keywords: list[str] = Field(
        ..., min_length=1, max_length=5, description="Korean adjective/noun keywords"
    )
    hero_features: list[str] = Field(
        ..., min_length=1, max_length=5, description="Selected Feature tag_en values"
    )
    narrative_structure: NarrativeStructure
    target_emotion: str = Field(..., description="e.g. curiosity -> admiration -> desire")
    tone_direction: str = Field(..., description="e.g. discovering a secret luxury spot")
    hook_direction: str = Field(..., description="e.g. 'Is this even in Korea?' style")
    cta_direction: str = Field(..., description="e.g. 'Check available dates now'")


class PlannedShot(BaseModel):
    """Single shot planned by the Producer/Director agent.

    Converts to StoryboardShot via to_storyboard_shot().
    """

    shot_id: int
    role: ShotRole
    duration_sec: float = Field(gt=0.0, le=10.0)
    feature_tag: str | None = None
    camera: CameraType = CameraType.STATIC
    image_index: int = 0
    visual_direction: str = ""
    transition: TransitionType = TransitionType.CUT

    def to_storyboard_shot(
        self,
        *,
        copy: ShotCopy,
        asset_path: str,
        claim_level: ClaimLevel | None = None,
        start_sec: float = 0.0,
        end_sec: float | None = None,
    ) -> "StoryboardShot":
        """Convert PlannedShot to StoryboardShot for final assembly."""
        from reels.production.models import StoryboardShot

        return StoryboardShot(
            shot_id=self.shot_id,
            role=self.role,
            start_sec=start_sec,
            end_sec=end_sec if end_sec is not None else start_sec + self.duration_sec,
            duration_sec=self.duration_sec,
            asset_type="image",
            asset_path=asset_path,
            feature_tag=self.feature_tag,
            claim_level=claim_level,
            place_label=self.feature_tag or "other",
            camera_suggestion=self.camera,
            copy=copy,
            edit=EditInfo(speed=1.0, transition_out=self.transition),
        )


class ShotPlan(BaseModel):
    """Complete shot plan from the Producer/Director agent."""

    shots: list[PlannedShot] = Field(..., min_length=1)
    total_duration_sec: float = Field(gt=0.0)
    pacing_note: str = ""
    music_mood: str = ""


class QAIssue(BaseModel):
    """Single issue found by the QA Reviewer.

    responsible_agent enables targeted revision routing.
    target_element uses JSONPath-style notation (e.g. "shots[2].caption_lines[0]").
    """

    severity: Literal["critical", "warning", "suggestion"]
    responsible_agent: Literal["planner", "director", "writer"]
    target_element: str
    current_value: str
    expected_behavior: str
    related_rule: str | None = None
    suggestion: str = ""


class QAReport(BaseModel):
    """QA review result from the Reviewer agent."""

    verdict: Literal["PASS", "REVISE"]
    issues: list[QAIssue] = []
    revision_count: int = 0
