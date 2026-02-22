"""ProducerDirector — shot planning agent for the creative team pipeline."""

from __future__ import annotations

import json
import logging
from typing import Any

from reels.production.creative_team.llm import CreativeLLM
from reels.production.creative_team.models import CreativeBrief, PlannedShot, QAIssue, ShotPlan
from reels.production.models import VerifiedFeature
from reels.models.template import Template

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = """\
당신은 숙소 마케팅 숏폼 영상 PD입니다.

기획자의 CreativeBrief를 바탕으로 구체적인 샷 구성을 결정합니다.

규칙:
1. 총 영상 길이: 10-15초
2. 샷 수: 5-8개
3. role은 "hook" | "feature" | "support" | "cta" 중 하나
4. camera는 "static" | "pan_left" | "pan_right" | "tilt_up" | "tilt_down" | "push_in" | "pull_out" | "handheld" | "gimbal_smooth" 중 하나
5. transition은 "cut" | "cut_on_beat" | "dissolve" | "fade" 중 하나
6. 첫 샷은 반드시 hook, 마지막 샷은 반드시 cta
7. hero_features의 각 feature가 최소 1개 샷에 배정되어야 함
8. image_index는 0부터 시작하는 입력 이미지 인덱스
9. 내러티브 구조에 따른 샷 순서:
   - reveal: 외관 → 내부 → 핵심 어메니티 (점진적 공개)
   - highlight: 베스트 Feature → 보조 → CTA (임팩트 우선)
   - story: 체크인 → 객실 → 식사 → 힐링 (동선 기반)
   - comparison: 평범한 외관 → 놀라운 내부 (반전)

JSON만 출력하세요.\
"""


def _build_user_prompt(
    brief: CreativeBrief,
    features: list[VerifiedFeature],
    assets: list[Any],
    template: Template | None,
) -> str:
    brief_json = brief.model_dump_json(indent=2)
    asset_count = len(assets)
    last_idx = max(asset_count - 1, 0)

    feature_lines = "\n".join(
        f"- {f.tag_en} (카테고리: {f.category}, 신뢰도: {f.confidence:.2f}, 레벨: {f.claim_level})"
        for f in features
    )

    parts = [
        f"콘셉트 브리프:\n{brief_json}",
        f"\n사용 가능한 이미지: {asset_count}장 (인덱스 0-{last_idx})",
        f"\nFeature 목록 ({len(features)}개):\n{feature_lines}",
    ]

    if template is not None:
        parts.append(
            f"\n참조 템플릿: {template.template_id} "
            f"({template.shot_count}샷, {template.total_duration_sec:.1f}초)"
        )

    parts.append("\n위 정보를 기반으로 ShotPlan을 JSON으로 작성하세요.")
    return "".join(parts)


def _clamp_image_indices(shot_plan: ShotPlan, max_index: int) -> ShotPlan:
    """Clamp each shot's image_index to [0, max_index] and recompute total_duration_sec."""
    clamped_shots: list[PlannedShot] = []
    for shot in shot_plan.shots:
        if shot.image_index < 0 or shot.image_index > max_index:
            clamped = shot.model_copy(update={"image_index": max(0, min(shot.image_index, max_index))})
        else:
            clamped = shot
        clamped_shots.append(clamped)

    total = sum(s.duration_sec for s in clamped_shots)
    return shot_plan.model_copy(update={"shots": clamped_shots, "total_duration_sec": total})


class ProducerDirector:
    """Shot planning agent.

    Converts a CreativeBrief into a concrete ShotPlan, and can revise
    an existing ShotPlan based on QAIssue feedback.
    """

    def __init__(self, llm: CreativeLLM, config: dict | None = None) -> None:
        cfg = (config or {}).get("production", {}).get("creative_team", {}).get("director", {})
        self._llm = llm
        self._model: str | None = cfg.get("model")
        self._temperature: float = cfg.get("temperature", 0.5)
        self._max_tokens: int = cfg.get("max_tokens", 2048)

    async def direct(
        self,
        brief: CreativeBrief,
        features: list[VerifiedFeature],
        assets: list[Any],
        template: Template | None = None,
    ) -> ShotPlan:
        """Generate a ShotPlan from a CreativeBrief.

        Args:
            brief: Creative concept document from the Planner agent.
            features: Verified features extracted from accommodation images.
            assets: List of image asset paths (used only for count/index range).
            template: Optional reference template from the DB.

        Returns:
            Validated ShotPlan with clamped image indices and recomputed total_duration_sec.
        """
        prompt = _build_user_prompt(brief, features, assets, template)

        logger.debug(
            "ProducerDirector.direct: %d features, %d assets, template=%s",
            len(features),
            len(assets),
            template.template_id if template else None,
        )

        shot_plan = await self._llm.generate(
            system=_DEFAULT_SYSTEM_PROMPT,
            prompt=prompt,
            response_model=ShotPlan,
            agent_role="director",
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        max_index = max(len(assets) - 1, 0)
        return _clamp_image_indices(shot_plan, max_index)

    async def revise(self, previous: ShotPlan, issues: list[QAIssue]) -> ShotPlan:
        """Revise a ShotPlan based on QA issues.

        Args:
            previous: The ShotPlan that failed QA review.
            issues: List of QAIssues targeting this agent (responsible_agent == "director").

        Returns:
            Revised ShotPlan with the same image-index clamping applied.
        """
        issues_json = json.dumps(
            [i.model_dump() for i in issues], ensure_ascii=False, indent=2
        )
        previous_json = previous.model_dump_json(indent=2)

        prompt = (
            "아래는 이전 ShotPlan입니다:\n"
            f"{previous_json}\n\n"
            "QA 검토 결과 다음 문제가 발견되었습니다:\n"
            f"{issues_json}\n\n"
            "문제가 되는 부분만 수정하여 개선된 ShotPlan을 JSON으로 출력하세요."
        )

        logger.debug(
            "ProducerDirector.revise: %d issue(s) to fix", len(issues)
        )

        # Derive asset count from the max image_index present in the previous plan
        max_existing = max((s.image_index for s in previous.shots), default=0)

        shot_plan = await self._llm.generate(
            system=_DEFAULT_SYSTEM_PROMPT,
            prompt=prompt,
            response_model=ShotPlan,
            agent_role="director",
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            use_cache=False,
        )

        return _clamp_image_indices(shot_plan, max_existing)
