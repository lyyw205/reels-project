"""CreativeWriter — generates per-shot copy from CreativeBrief + ShotPlan."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel

from reels.production.creative_team.llm import CreativeLLM
from reels.production.creative_team.models import CreativeBrief, QAIssue, ShotPlan
from reels.production.models import AccommodationInput, CaptionLine, ShotCopy, VerifiedFeature

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = """\
당신은 숙소 마케팅 숏폼 영상 카피라이터입니다.

PD의 ShotPlan에 맞춰 각 샷의 카피를 작성합니다.

카피 톤 규칙 (ClaimLevel 기반):
- CONFIRMED (신뢰도 >= 0.75): 단정적 표현 가능 ("노천탕 있는 프라이빗 힐링")
- PROBABLE (0.50-0.75): 암시적 표현 ("노천탕 느낌의 야외 욕실")
- SUGGESTIVE (< 0.50): 분위기 표현만 ("특별한 힐링 공간")

금지 단어 규칙:
- "무료", "제공", "무제한", "포함", "운영"은 CONFIRMED 레벨에서만 사용 가능
- PROBABLE/SUGGESTIVE에서 이 단어 사용 시 반드시 대체 표현 사용

자막 규칙:
- hook_line: 7-12 한글 글자
- caption 한 줄: 최대 14 한글 글자
- caption 최대 2줄
- CTA 캡션: 최대 14 한글 글자

JSON 객체로 출력. copies 배열은 shots 순서대로 1:1 대응.\
"""


class _CopiesResponse(BaseModel):
    copies: list[ShotCopy]


class CreativeWriter:
    """Generates per-shot copy from a CreativeBrief and ShotPlan.

    Reads config from production.creative_team.writer.
    """

    def __init__(self, llm: CreativeLLM, config: dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("production", {}).get("creative_team", {}).get("writer", {})
        self._llm = llm
        self._model: str | None = cfg.get("model")
        self._temperature: float = cfg.get("temperature", 0.9)
        self._max_tokens: int = cfg.get("max_tokens", 1536)
        self._system_prompt: str = cfg.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)

    async def write(
        self,
        brief: CreativeBrief,
        shot_plan: ShotPlan,
        features: list[VerifiedFeature],
        context: AccommodationInput,
    ) -> list[ShotCopy]:
        """Generate per-shot copy for each shot in the plan.

        Args:
            brief: Creative concept from the Planner agent.
            shot_plan: Shot structure from the Director agent.
            features: Verified features for copy tone guidance.
            context: Accommodation input with name, region, target, etc.

        Returns:
            List of ShotCopy, one per shot in shot_plan.shots.
            Length is guaranteed to equal len(shot_plan.shots).
        """
        prompt = self._build_write_prompt(brief, shot_plan, features, context)
        logger.debug(
            "CreativeWriter.write: %d shots, target=%s",
            len(shot_plan.shots),
            context.target_audience,
        )
        result = await self._llm.generate(
            system=self._system_prompt,
            prompt=prompt,
            response_model=_CopiesResponse,
            agent_role="writer",
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            use_cache=False,
        )
        return self._align_copies(result.copies, len(shot_plan.shots))

    async def revise(
        self,
        previous: list[ShotCopy],
        shot_plan: ShotPlan,
        issues: list[QAIssue],
    ) -> list[ShotCopy]:
        """Revise copies based on QA issues flagged for the writer.

        Only writer-owned issues are sent to the LLM. Non-writer issues are
        filtered out before building the prompt.

        Args:
            previous: Current copies to revise.
            shot_plan: Shot plan for shot count alignment.
            issues: All QA issues (writer-relevant ones are filtered here).

        Returns:
            Revised list of ShotCopy, length aligned to shot_plan.shots.
        """
        writer_issues = [i for i in issues if i.responsible_agent == "writer"]
        if not writer_issues:
            logger.debug("CreativeWriter.revise: no writer issues, returning previous copies")
            return self._align_copies(previous, len(shot_plan.shots))

        issue_lines = "\n".join(
            f"- [{issue.severity}] {issue.target_element}: {issue.expected_behavior}"
            + (f" (제안: {issue.suggestion})" if issue.suggestion else "")
            for issue in writer_issues
        )

        previous_json = json.dumps(
            [copy.model_dump() for copy in previous],
            ensure_ascii=False,
            indent=2,
        )

        prompt = (
            "다음 copies에서 아래 이슈만 수정하세요. 나머지는 그대로 유지하세요.\n\n"
            f"기존 copies ({len(previous)}개):\n{previous_json}\n\n"
            f"수정할 이슈:\n{issue_lines}\n\n"
            f"수정된 {len(shot_plan.shots)}개의 ShotCopy를 JSON으로 출력하세요.\n"
            "copies 배열의 각 항목은 shots 순서대로 대응합니다."
        )

        logger.debug("CreativeWriter.revise: %d writer issues to fix", len(writer_issues))
        result = await self._llm.generate(
            system=self._system_prompt,
            prompt=prompt,
            response_model=_CopiesResponse,
            agent_role="writer",
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            use_cache=False,
        )
        return self._align_copies(result.copies, len(shot_plan.shots))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_write_prompt(
        self,
        brief: CreativeBrief,
        shot_plan: ShotPlan,
        features: list[VerifiedFeature],
        context: AccommodationInput,
    ) -> str:
        brief_json = brief.model_dump_json(indent=2)

        shot_lines = "\n".join(
            f"- shot_{shot.shot_id}: role={shot.role.value}, "
            f"feature={shot.feature_tag or 'none'}, duration={shot.duration_sec}s"
            for shot in shot_plan.shots
        )

        # Collect feature tags referenced in shots
        used_tags: set[str] = {
            shot.feature_tag for shot in shot_plan.shots if shot.feature_tag is not None
        }
        feature_map = {f.tag_en: f for f in features}
        feature_lines = "\n".join(
            f"- {tag}: claim_level={feature_map[tag].claim_level.value}, "
            f"copy_tone={feature_map[tag].copy_tone}"
            for tag in used_tags
            if tag in feature_map
        )

        name = context.name or "미정"
        region = context.region or "미정"
        target = context.target_audience.value
        shot_count = len(shot_plan.shots)

        return (
            f"콘셉트 브리프:\n{brief_json}\n\n"
            f"ShotPlan ({shot_count}샷):\n{shot_lines}\n\n"
            f"Feature 상세:\n{feature_lines}\n\n"
            f"숙소: {name} ({region})\n"
            f"타겟: {target}\n\n"
            f"위 ShotPlan에 맞춰 {shot_count}개의 ShotCopy를 JSON으로 작성하세요.\n"
            "copies 배열의 각 항목은 shots 순서대로 대응합니다."
        )

    def _align_copies(self, copies: list[ShotCopy], expected: int) -> list[ShotCopy]:
        """Ensure copies list length matches expected shot count.

        Pads with empty ShotCopy() if shorter; truncates if longer.
        """
        if len(copies) < expected:
            logger.warning(
                "CreativeWriter: got %d copies but expected %d — padding with empty ShotCopy",
                len(copies),
                expected,
            )
            copies = copies + [ShotCopy() for _ in range(expected - len(copies))]
        elif len(copies) > expected:
            logger.warning(
                "CreativeWriter: got %d copies but expected %d — truncating",
                len(copies),
                expected,
            )
            copies = copies[:expected]
        return copies
