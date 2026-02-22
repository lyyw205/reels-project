"""Pydantic v2 models for the shorts production pipeline."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from reels.models.template import CameraType, EditInfo, TemplateFormat


class TargetAudience(StrEnum):
    COUPLE = "couple"
    FAMILY = "family"
    SOLO = "solo"
    FRIENDS = "friends"


class PriceRange(StrEnum):
    BUDGET = "budget"
    MID = "mid"
    LUXURY = "luxury"


class FeatureCategory(StrEnum):
    SCENE = "scene"       # 객실, 로비, 외관
    AMENITY = "amenity"   # 노천탕, 사우나, 풀
    VIEW = "view"         # 오션뷰, 마운틴뷰
    DINING = "dining"     # 조식, 뷔페, 바베큐
    ACTIVITY = "activity" # 키즈, 반려견, 불멍


class ClaimLevel(StrEnum):
    CONFIRMED = "confirmed"     # confidence >= 0.75 or web-verified
    PROBABLE = "probable"       # 0.50 <= confidence < 0.75
    SUGGESTIVE = "suggestive"   # confidence < 0.50


class ShotRole(StrEnum):
    HOOK = "hook"
    FEATURE = "feature"
    SUPPORT = "support"
    CTA = "cta"


class AccommodationInput(BaseModel):
    name: str | None = None
    region: str | None = None
    target_audience: TargetAudience = TargetAudience.COUPLE
    price_range: PriceRange | None = None
    images: list[Path]                    # required
    video_clips: list[Path] = []
    custom_instructions: str | None = None


class Feature(BaseModel):
    tag: str                              # Korean tag e.g. "노천탕"
    tag_en: str                           # English tag e.g. "outdoor_bath"
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_images: list[str] = []       # filenames
    description: str = ""                 # VLM reasoning
    category: FeatureCategory = FeatureCategory.SCENE


class WebEvidence(BaseModel):
    claim: str
    url: str
    snippet: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class VerifiedFeature(Feature):
    claim_level: ClaimLevel = ClaimLevel.SUGGESTIVE
    web_evidence: list[WebEvidence] = []
    copy_tone: str = "분위기"             # "단정" / "암시" / "분위기"


class CaptionLine(BaseModel):
    text: str
    start_sec: float = 0.0
    end_sec: float = 0.0
    position: str = "bottom_center"


class ShotCopy(BaseModel):
    hook_line: str | None = None          # 7-12 chars
    caption_lines: list[CaptionLine] = []
    vo_script: str | None = None


class StoryboardShot(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    shot_id: int
    role: ShotRole
    start_sec: float
    end_sec: float
    duration_sec: float
    asset_type: str = "image"             # "image" | "video_clip"
    asset_path: str = ""
    feature_tag: str | None = None
    claim_level: ClaimLevel | None = None
    place_label: str = "other"
    camera_suggestion: CameraType = CameraType.STATIC
    shot_copy: ShotCopy = Field(
        default_factory=ShotCopy,
        validation_alias=AliasChoices("copy", "shot_copy"),
        serialization_alias="copy",
    )
    edit: EditInfo = Field(default_factory=EditInfo)


class Storyboard(BaseModel):
    project_id: str
    accommodation_name: str | None = None
    target_audience: TargetAudience = TargetAudience.COUPLE
    features: list[VerifiedFeature] = []
    template_ref: str | None = None       # reference template ID
    total_duration_sec: float = 12.0
    shots: list[StoryboardShot] = []


class AssetTransform(BaseModel):
    resize_to: tuple[int, int] = (1080, 1920)
    crop_mode: str = "center"
    speed: float = 1.0
    ken_burns: bool = False
    ken_burns_direction: str | None = None


class AssetMapping(BaseModel):
    shot_id: int
    source_path: str
    transform: AssetTransform = Field(default_factory=AssetTransform)


class MusicSpec(BaseModel):
    source: str = "auto"
    bpm_target: float | None = None
    volume: float = 0.3
    fade_in_sec: float = 0.5
    fade_out_sec: float = 1.0


class RenderSpec(BaseModel):
    project_id: str
    format: TemplateFormat = Field(default_factory=TemplateFormat)
    storyboard: Storyboard
    assets: list[AssetMapping] = []
    captions_srt: str = ""
    vo_script: str | None = None
    music: MusicSpec | None = None
    output_path: str = ""


class MatchResult(BaseModel):
    template_id: str
    score: float
    shot_count: int
    template_duration: float


class ProductionResult(BaseModel):
    project_id: str
    status: str = "complete"              # "complete" | "partial" | "failed"
    storyboard: Storyboard | None = None
    render_spec: RenderSpec | None = None
    features: list[VerifiedFeature] = []
    errors: list[str] = []
    output_dir: str = ""
