"""Deterministic QA code checks for storyboard validation.

Extracted from ``reels.production.creative_team.reviewer._run_code_checks``
so that both the Python QAReviewer class and OMC reviewer agent can reuse
the same logic.

CLI usage::

    python -m reels.production.omc_helpers.code_checks <storyboard.json> <features.json>

Outputs a JSON array of QAIssue dicts to stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from reels.production.creative_team.models import QAIssue
from reels.production.models import ClaimLevel, Storyboard, VerifiedFeature

_FACTUAL_WORDS = frozenset(["무료", "제공", "무제한", "포함", "운영"])


def run_code_checks(
    storyboard: Storyboard,
    features: list[VerifiedFeature],
) -> list[QAIssue]:
    """Run deterministic rule checks on a storyboard. No LLM involved.

    Checks:
    - Factual words in non-CONFIRMED copy
    - hook_line char limit (max 12)
    - caption_lines char limit (max 14)
    - Total duration range (10-15s)
    - First shot must be hook, last shot must be cta
    """
    issues: list[QAIssue] = []
    feature_map = {f.tag_en: f for f in features}

    for i, shot in enumerate(storyboard.shots):
        feat = feature_map.get(shot.feature_tag) if shot.feature_tag else None
        claim_level = feat.claim_level if feat else None

        # Check factual words in non-CONFIRMED copy
        if claim_level and claim_level != ClaimLevel.CONFIRMED:
            texts_to_check: list[tuple[str, str]] = []
            if shot.shot_copy.hook_line:
                texts_to_check.append(("hook_line", shot.shot_copy.hook_line))
            for j, cap in enumerate(shot.shot_copy.caption_lines):
                texts_to_check.append((f"caption_lines[{j}]", cap.text))

            for field, text in texts_to_check:
                for word in _FACTUAL_WORDS:
                    if word in text:
                        issues.append(
                            QAIssue(
                                severity="critical",
                                responsible_agent="writer",
                                target_element=f"shots[{i}].copy.{field}",
                                current_value=text,
                                expected_behavior=f"{claim_level} 레벨에서 '{word}' 사용 불가",
                                related_rule="factual_words_confirmed_only",
                                suggestion=f"'{word}'를 대체 표현으로 변경",
                            )
                        )

        # Check hook_line char limit
        if shot.shot_copy.hook_line and len(shot.shot_copy.hook_line) > 12:
            issues.append(
                QAIssue(
                    severity="warning",
                    responsible_agent="writer",
                    target_element=f"shots[{i}].copy.hook_line",
                    current_value=shot.shot_copy.hook_line,
                    expected_behavior="hook_line 최대 12자",
                    related_rule="hook_max_chars",
                )
            )

        # Check caption char limits
        for j, cap in enumerate(shot.shot_copy.caption_lines):
            if len(cap.text) > 14:
                issues.append(
                    QAIssue(
                        severity="warning",
                        responsible_agent="writer",
                        target_element=f"shots[{i}].copy.caption_lines[{j}]",
                        current_value=cap.text,
                        expected_behavior="caption 최대 14자",
                        related_rule="caption_max_chars",
                    )
                )

    # Check total duration range
    total = sum(s.duration_sec for s in storyboard.shots)
    if total < 10.0 or total > 15.0:
        issues.append(
            QAIssue(
                severity="critical",
                responsible_agent="director",
                target_element="total_duration_sec",
                current_value=f"{total:.1f}",
                expected_behavior="총 길이 10-15초",
                related_rule="duration_range",
            )
        )

    # Check structure: first=hook, last=cta
    if storyboard.shots and storyboard.shots[0].role != "hook":
        issues.append(
            QAIssue(
                severity="critical",
                responsible_agent="director",
                target_element="shots[0].role",
                current_value=storyboard.shots[0].role,
                expected_behavior="첫 샷은 hook",
                related_rule="structure_first_hook",
            )
        )
    if storyboard.shots and storyboard.shots[-1].role != "cta":
        issues.append(
            QAIssue(
                severity="critical",
                responsible_agent="director",
                target_element=f"shots[{len(storyboard.shots) - 1}].role",
                current_value=storyboard.shots[-1].role,
                expected_behavior="마지막 샷은 cta",
                related_rule="structure_last_cta",
            )
        )

    return issues


def main() -> None:
    """CLI entry point: reads storyboard.json + features.json, prints issues."""
    if len(sys.argv) < 3:
        print(
            "Usage: python -m reels.production.omc_helpers.code_checks "
            "<storyboard.json> <features.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    storyboard_path = Path(sys.argv[1])
    features_path = Path(sys.argv[2])

    storyboard = Storyboard.model_validate_json(storyboard_path.read_text())
    features_data = json.loads(features_path.read_text())
    features = [VerifiedFeature.model_validate(f) for f in features_data]

    issues = run_code_checks(storyboard, features)

    result = [issue.model_dump() for issue in issues]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
