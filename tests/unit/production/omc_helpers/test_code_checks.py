"""Tests for reels.production.omc_helpers.code_checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from reels.production.creative_team.models import QAIssue
from reels.production.models import (
    CaptionLine,
    ClaimLevel,
    ShotCopy,
    ShotRole,
    Storyboard,
    StoryboardShot,
    VerifiedFeature,
)
from reels.production.omc_helpers.code_checks import run_code_checks


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_feature(tag_en: str, claim_level: ClaimLevel = ClaimLevel.CONFIRMED) -> VerifiedFeature:
    return VerifiedFeature(
        tag="테스트",
        tag_en=tag_en,
        confidence=0.9 if claim_level == ClaimLevel.CONFIRMED else 0.6,
        claim_level=claim_level,
        category="amenity",
    )


def _make_shot(
    shot_id: int,
    role: str = "feature",
    duration: float = 2.0,
    feature_tag: str | None = "outdoor_bath",
    hook_line: str | None = None,
    captions: list[str] | None = None,
) -> StoryboardShot:
    caption_lines = [CaptionLine(text=t) for t in (captions or [])]
    return StoryboardShot(
        shot_id=shot_id,
        role=ShotRole(role),
        start_sec=0.0,
        end_sec=duration,
        duration_sec=duration,
        feature_tag=feature_tag,
        copy=ShotCopy(hook_line=hook_line, caption_lines=caption_lines),
    )


def _make_storyboard(shots: list[StoryboardShot]) -> Storyboard:
    total = sum(s.duration_sec for s in shots)
    return Storyboard(
        project_id="test_001",
        shots=shots,
        total_duration_sec=total,
    )


# ── Factual word tests ───────────────────────────────────────────────


class TestFactualWords:
    def test_probable_with_factual_word_is_critical(self):
        feat = _make_feature("outdoor_bath", ClaimLevel.PROBABLE)
        shot = _make_shot(0, role="hook", hook_line="무료 조식 제공", feature_tag="outdoor_bath")
        sb = _make_storyboard([shot, _make_shot(1, role="cta", duration=10.0, feature_tag=None)])
        issues = run_code_checks(sb, [feat])
        factual_issues = [i for i in issues if i.related_rule == "factual_words_confirmed_only"]
        assert len(factual_issues) >= 1
        assert all(i.severity == "critical" for i in factual_issues)

    def test_confirmed_with_factual_word_no_issue(self):
        feat = _make_feature("outdoor_bath", ClaimLevel.CONFIRMED)
        shot = _make_shot(0, role="hook", hook_line="무료 제공", feature_tag="outdoor_bath")
        sb = _make_storyboard([shot, _make_shot(1, role="cta", duration=10.0, feature_tag=None)])
        issues = run_code_checks(sb, [feat])
        factual_issues = [i for i in issues if i.related_rule == "factual_words_confirmed_only"]
        assert len(factual_issues) == 0

    def test_probable_caption_with_factual_word(self):
        feat = _make_feature("pool", ClaimLevel.PROBABLE)
        shot = _make_shot(0, role="hook", captions=["무제한 이용"], feature_tag="pool")
        sb = _make_storyboard([shot, _make_shot(1, role="cta", duration=10.0, feature_tag=None)])
        issues = run_code_checks(sb, [feat])
        factual_issues = [i for i in issues if i.related_rule == "factual_words_confirmed_only"]
        assert len(factual_issues) >= 1

    def test_no_feature_tag_skips_factual_check(self):
        shot = _make_shot(0, role="hook", hook_line="무료 숙소", feature_tag=None)
        sb = _make_storyboard([shot, _make_shot(1, role="cta", duration=10.0, feature_tag=None)])
        issues = run_code_checks(sb, [])
        factual_issues = [i for i in issues if i.related_rule == "factual_words_confirmed_only"]
        assert len(factual_issues) == 0


# ── Hook char limit tests ────────────────────────────────────────────


class TestHookCharLimit:
    def test_hook_over_12_chars_is_warning(self):
        shot = _make_shot(0, role="hook", hook_line="이것은열세글자가넘는훅라인입니다", feature_tag=None)
        sb = _make_storyboard([shot, _make_shot(1, role="cta", duration=10.0, feature_tag=None)])
        issues = run_code_checks(sb, [])
        hook_issues = [i for i in issues if i.related_rule == "hook_max_chars"]
        assert len(hook_issues) == 1
        assert hook_issues[0].severity == "warning"

    def test_hook_exactly_12_chars_no_issue(self):
        shot = _make_shot(0, role="hook", hook_line="가나다라마바사아자차카타", feature_tag=None)
        assert len("가나다라마바사아자차카타") == 12
        sb = _make_storyboard([shot, _make_shot(1, role="cta", duration=10.0, feature_tag=None)])
        issues = run_code_checks(sb, [])
        hook_issues = [i for i in issues if i.related_rule == "hook_max_chars"]
        assert len(hook_issues) == 0


# ── Caption char limit tests ─────────────────────────────────────────


class TestCaptionCharLimit:
    def test_caption_over_14_chars_is_warning(self):
        shot = _make_shot(0, role="hook", captions=["이것은열다섯글자가넘는캡션문장입니다"], feature_tag=None)
        sb = _make_storyboard([shot, _make_shot(1, role="cta", duration=10.0, feature_tag=None)])
        issues = run_code_checks(sb, [])
        cap_issues = [i for i in issues if i.related_rule == "caption_max_chars"]
        assert len(cap_issues) == 1
        assert cap_issues[0].severity == "warning"

    def test_caption_within_14_chars_no_issue(self):
        shot = _make_shot(0, role="hook", captions=["열네글자이내캡션"], feature_tag=None)
        sb = _make_storyboard([shot, _make_shot(1, role="cta", duration=10.0, feature_tag=None)])
        issues = run_code_checks(sb, [])
        cap_issues = [i for i in issues if i.related_rule == "caption_max_chars"]
        assert len(cap_issues) == 0


# ── Duration tests ────────────────────────────────────────────────────


class TestDuration:
    def test_under_10s_is_critical(self):
        shots = [
            _make_shot(0, role="hook", duration=2.0, feature_tag=None),
            _make_shot(1, role="cta", duration=2.0, feature_tag=None),
        ]
        sb = _make_storyboard(shots)
        issues = run_code_checks(sb, [])
        dur_issues = [i for i in issues if i.related_rule == "duration_range"]
        assert len(dur_issues) == 1
        assert dur_issues[0].severity == "critical"

    def test_over_15s_is_critical(self):
        shots = [
            _make_shot(0, role="hook", duration=8.0, feature_tag=None),
            _make_shot(1, role="cta", duration=8.0, feature_tag=None),
        ]
        sb = _make_storyboard(shots)
        issues = run_code_checks(sb, [])
        dur_issues = [i for i in issues if i.related_rule == "duration_range"]
        assert len(dur_issues) == 1
        assert dur_issues[0].severity == "critical"

    def test_within_range_no_issue(self):
        shots = [
            _make_shot(0, role="hook", duration=2.0, feature_tag=None),
            _make_shot(1, role="feature", duration=6.0, feature_tag=None),
            _make_shot(2, role="cta", duration=4.0, feature_tag=None),
        ]
        sb = _make_storyboard(shots)
        issues = run_code_checks(sb, [])
        dur_issues = [i for i in issues if i.related_rule == "duration_range"]
        assert len(dur_issues) == 0


# ── Structure tests ───────────────────────────────────────────────────


class TestStructure:
    def test_first_not_hook_is_critical(self):
        shots = [
            _make_shot(0, role="feature", duration=5.0, feature_tag=None),
            _make_shot(1, role="cta", duration=7.0, feature_tag=None),
        ]
        sb = _make_storyboard(shots)
        issues = run_code_checks(sb, [])
        struct_issues = [i for i in issues if i.related_rule == "structure_first_hook"]
        assert len(struct_issues) == 1
        assert struct_issues[0].severity == "critical"

    def test_last_not_cta_is_critical(self):
        shots = [
            _make_shot(0, role="hook", duration=5.0, feature_tag=None),
            _make_shot(1, role="feature", duration=7.0, feature_tag=None),
        ]
        sb = _make_storyboard(shots)
        issues = run_code_checks(sb, [])
        struct_issues = [i for i in issues if i.related_rule == "structure_last_cta"]
        assert len(struct_issues) == 1
        assert struct_issues[0].severity == "critical"

    def test_valid_structure_no_issue(self):
        shots = [
            _make_shot(0, role="hook", duration=2.0, feature_tag=None),
            _make_shot(1, role="feature", duration=6.0, feature_tag=None),
            _make_shot(2, role="cta", duration=4.0, feature_tag=None),
        ]
        sb = _make_storyboard(shots)
        issues = run_code_checks(sb, [])
        struct_issues = [
            i for i in issues
            if i.related_rule in ("structure_first_hook", "structure_last_cta")
        ]
        assert len(struct_issues) == 0


# ── CLI entry point test ─────────────────────────────────────────────


class TestCLI:
    def test_cli_runs_and_outputs_json(self, tmp_path: Path):
        feat = _make_feature("pool", ClaimLevel.PROBABLE)
        shot = _make_shot(0, role="hook", hook_line="무료 수영장", feature_tag="pool")
        cta = _make_shot(1, role="cta", duration=10.0, feature_tag=None)
        sb = _make_storyboard([shot, cta])

        sb_path = tmp_path / "storyboard.json"
        sb_path.write_text(sb.model_dump_json(indent=2))

        feat_path = tmp_path / "features.json"
        feat_path.write_text(json.dumps([feat.model_dump(mode="json")], ensure_ascii=False))

        result = subprocess.run(
            [
                sys.executable, "-m", "reels.production.omc_helpers.code_checks",
                str(sb_path), str(feat_path),
            ],
            capture_output=True,
            text=True,
            cwd="/home/youngwoo/repos/reels-project",
        )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert isinstance(output, list)
        assert len(output) >= 1  # at least the factual word issue
