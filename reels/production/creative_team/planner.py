"""CreativePlanner — generates CreativeBrief from accommodation features."""

from __future__ import annotations

import logging
from typing import Any

from reels.production.creative_team.llm import CreativeLLM
from reels.production.creative_team.models import CreativeBrief, QAIssue
from reels.production.models import AccommodationInput, ClaimLevel, MatchResult, VerifiedFeature

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = """\
당신은 숙소 마케팅 숏폼 영상 기획 전문가입니다.

숙소의 Feature 목록과 타겟 오디언스를 분석하여 영상 콘셉트를 설계합니다.

규칙:
1. hero_features는 반드시 입력된 Feature tag_en 목록에서 3개를 선택할 것 (부족하면 있는 만큼)
2. narrative_structure는 "reveal" | "highlight" | "story" | "comparison" 중 하나
3. concept_keywords는 한국어 형용사/명사 3-5개
4. 타겟별 소구점:
   - couple: 프라이빗, 로맨틱, 둘만의
   - family: 안심, 함께, 편안한
   - solo: 여유, 나만의, 힐링
   - friends: 신나는, 함께, 인생샷

JSON만 출력하세요. 설명 없이 JSON 객체만.\
"""


class CreativePlanner:
    """Generates a CreativeBrief from accommodation features and context.

    Reads config from production.creative_team.planner.
    """

    def __init__(self, llm: CreativeLLM, config: dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("production", {}).get("creative_team", {}).get("planner", {})
        self._llm = llm
        self._model: str | None = cfg.get("model")
        self._temperature: float = cfg.get("temperature", 0.8)
        self._max_tokens: int = cfg.get("max_tokens", 1024)
        self._system_prompt: str = cfg.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)

    async def plan(
        self,
        features: list[VerifiedFeature],
        context: AccommodationInput,
        match_result: MatchResult | None = None,
    ) -> CreativeBrief:
        """Generate a CreativeBrief from features and accommodation context.

        Args:
            features: Verified features extracted from accommodation images.
            context: Accommodation input with name, region, target, etc.
            match_result: Optional matched template info to include in prompt.

        Returns:
            Structured CreativeBrief with concept, hero features, and directions.
        """
        prompt = self._build_plan_prompt(features, context, match_result)
        logger.debug(
            "CreativePlanner.plan: %d features, target=%s",
            len(features),
            context.target_audience,
        )
        return await self._llm.generate(
            system=self._system_prompt,
            prompt=prompt,
            response_model=CreativeBrief,
            agent_role="planner",
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

    async def revise(
        self,
        previous: CreativeBrief,
        issues: list[QAIssue],
    ) -> CreativeBrief:
        """Revise a CreativeBrief based on QA issues.

        Only planner-owned issues are expected here; the method instructs the
        LLM to fix those issues while preserving the rest of the brief.

        Args:
            previous: The brief to revise.
            issues: QAIssues that require changes to the planner output.

        Returns:
            Revised CreativeBrief.
        """
        issue_lines = "\n".join(
            f"- [{issue.severity}] {issue.target_element}: {issue.expected_behavior}"
            + (f" (제안: {issue.suggestion})" if issue.suggestion else "")
            for issue in issues
        )
        prompt = (
            "다음 CreativeBrief에서 아래 이슈만 수정하세요. 나머지는 그대로 유지하세요.\n\n"
            f"기존 CreativeBrief:\n{previous.model_dump_json(indent=2)}\n\n"
            f"수정할 이슈:\n{issue_lines}\n\n"
            "수정된 CreativeBrief를 JSON으로 출력하세요."
        )
        logger.debug("CreativePlanner.revise: %d issues to fix", len(issues))
        return await self._llm.generate(
            system=self._system_prompt,
            prompt=prompt,
            response_model=CreativeBrief,
            agent_role="planner",
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_plan_prompt(
        self,
        features: list[VerifiedFeature],
        context: AccommodationInput,
        match_result: MatchResult | None,
    ) -> str:
        name = context.name or "미정"
        region = context.region or "미정"
        price_range = context.price_range.value if context.price_range else "미정"
        target = context.target_audience.value

        feature_lines = "\n".join(
            f"- {f.tag_en} ({f.category.value}, 신뢰도: {f.confidence:.2f}, 레벨: {f.claim_level.value})"
            for f in features
        )

        lines: list[str] = [
            f"숙소: {name} ({region})",
            f"타겟: {target}",
            f"가격대: {price_range}",
            "",
            f"발견된 Feature ({len(features)}개):",
            feature_lines,
        ]

        if match_result is not None:
            lines.append(
                f"\n매칭 템플릿: {match_result.template_id} "
                f"({match_result.shot_count}샷, {match_result.template_duration:.1f}초)"
            )

        if context.custom_instructions:
            lines.append(f"\n커스텀 지시: {context.custom_instructions}")

        lines.append("\n위 정보를 기반으로 CreativeBrief를 JSON으로 작성하세요.")

        return "\n".join(lines)
