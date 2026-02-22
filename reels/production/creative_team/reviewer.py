"""QAReviewer — storyboard quality assurance agent for the creative team pipeline."""

from __future__ import annotations

import logging

from reels.production.creative_team.llm import CreativeLLM
from reels.production.creative_team.models import QAIssue, QAReport
from reels.production.models import Storyboard, VerifiedFeature
from reels.production.omc_helpers.code_checks import run_code_checks

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = """\
당신은 숙소 마케팅 숏폼 영상 QA 검수자입니다.

완성된 스토리보드를 검토하고 문제를 지적합니다.

검증 항목:
1. 팩트체크: 각 샷의 카피가 해당 Feature의 ClaimLevel 톤 규칙을 준수하는지
2. 금지 단어: "무료/제공/무제한/포함/운영"이 CONFIRMED 아닌 Feature에 사용되었는지
3. 글자수: hook 7-12자, caption 14자 이내, 최대 2줄
4. 총 길이: 10-15초
5. 구조: 첫 샷 hook, 마지막 샷 cta
6. 흐름: 내러티브 일관성, 훅→CTA 자연스러운 연결

판정:
- 모든 검증 통과 → verdict: "PASS"
- critical 이슈 1개 이상 → verdict: "REVISE"
- warning만 있으면 → verdict: "PASS" (warning 기록)

각 이슈에 responsible_agent와 target_element를 명시하세요.

JSON만 출력하세요.\
"""


def _build_user_prompt(storyboard: Storyboard, features: list[VerifiedFeature]) -> str:
    storyboard_json = storyboard.model_dump_json(indent=2)
    feature_lines = "\n".join(
        f"- {f.tag_en} (태그: {f.tag}, 레벨: {f.claim_level}, 신뢰도: {f.confidence:.2f})"
        for f in features
    )
    return (
        f"스토리보드:\n{storyboard_json}\n\n"
        f"Feature 목록 ({len(features)}개):\n{feature_lines}\n\n"
        "위 스토리보드의 주관적 품질(내러티브 흐름, 훅→CTA 연결, 감정 일관성)을 검토하고 "
        "QAReport를 JSON으로 출력하세요. "
        "코드 규칙(글자수, 금지어, 길이, 구조)은 이미 별도 검사됩니다 — "
        "여기서는 흐름/일관성/설득력 등 주관적 항목만 평가하세요."
    )


class QAReviewer:
    """QA review agent.

    Combines deterministic code checks with LLM-based subjective review
    to produce a final QAReport with PASS/REVISE verdict.
    """

    def __init__(self, llm: CreativeLLM, config: dict | None = None) -> None:
        cfg = (config or {}).get("production", {}).get("creative_team", {}).get("reviewer", {})
        self._llm = llm
        self._model: str | None = cfg.get("model")
        self._temperature: float = cfg.get("temperature", 0.2)
        self._max_tokens: int = cfg.get("max_tokens", 1024)

    async def review(
        self,
        storyboard: Storyboard,
        features: list[VerifiedFeature],
        revision_count: int = 0,
    ) -> QAReport:
        """Review a completed storyboard and return a QAReport.

        Runs deterministic code checks first, then asks the LLM for
        subjective quality issues (narrative flow, consistency). Merges
        both sets of issues and derives the final verdict.

        Args:
            storyboard: The completed storyboard to review.
            features: Verified features used to build the storyboard.
            revision_count: Number of revisions already performed.

        Returns:
            QAReport with verdict PASS or REVISE and full issue list.
        """
        # Hard deterministic checks (no LLM required)
        code_issues = self._run_code_checks(storyboard, features)

        # Subjective LLM review (narrative flow, consistency, persuasiveness)
        llm_report = await self._llm_review(storyboard, features)

        # Merge and re-derive verdict based on combined issues
        all_issues = code_issues + llm_report.issues
        has_critical = any(i.severity == "critical" for i in all_issues)
        from typing import Literal
        verdict: Literal["PASS", "REVISE"] = "REVISE" if has_critical else "PASS"

        logger.debug(
            "QAReviewer.review: %d code issue(s), %d llm issue(s), verdict=%s",
            len(code_issues),
            len(llm_report.issues),
            verdict,
        )

        return QAReport(verdict=verdict, issues=all_issues, revision_count=revision_count)

    def _run_code_checks(
        self, storyboard: Storyboard, features: list[VerifiedFeature]
    ) -> list[QAIssue]:
        """Run deterministic rule checks. No LLM involved.

        Delegates to the shared ``run_code_checks`` function in
        ``reels.production.omc_helpers.code_checks`` for DRY reuse.
        """
        return run_code_checks(storyboard, features)

    async def _llm_review(
        self, storyboard: Storyboard, features: list[VerifiedFeature]
    ) -> QAReport:
        """Ask LLM to evaluate subjective quality (flow, narrative consistency)."""
        prompt = _build_user_prompt(storyboard, features)

        logger.debug(
            "QAReviewer._llm_review: storyboard %s, %d shot(s)",
            storyboard.project_id,
            len(storyboard.shots),
        )

        return await self._llm.generate(
            system=_DEFAULT_SYSTEM_PROMPT,
            prompt=prompt,
            response_model=QAReport,
            agent_role="reviewer",
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            use_cache=True,
        )
