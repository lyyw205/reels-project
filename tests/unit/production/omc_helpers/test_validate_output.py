"""Tests for reels.production.omc_helpers.validate_output."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from reels.production.omc_helpers.validate_output import validate_output, VALID_TYPES


# ── validate_output function tests ────────────────────────────────────


class TestValidateOutput:
    def test_valid_brief(self):
        data = {
            "concept_keywords": ["로맨틱", "프라이빗"],
            "hero_features": ["outdoor_bath"],
            "narrative_structure": "reveal",
            "target_emotion": "curiosity -> admiration -> desire",
            "tone_direction": "비밀스러운 럭셔리 발견",
            "hook_direction": "'이게 국내야?' 스타일",
            "cta_direction": "'빈 날짜 확인하기'",
        }
        errors = validate_output("brief", data)
        assert errors == []

    def test_invalid_brief_missing_fields(self):
        data = {"concept_keywords": ["로맨틱"]}
        errors = validate_output("brief", data)
        assert len(errors) > 0

    def test_invalid_brief_empty_keywords(self):
        data = {
            "concept_keywords": [],  # min_length=1
            "hero_features": ["outdoor_bath"],
            "narrative_structure": "reveal",
            "target_emotion": "test",
            "tone_direction": "test",
            "hook_direction": "test",
            "cta_direction": "test",
        }
        errors = validate_output("brief", data)
        assert len(errors) > 0

    def test_valid_shot_plan(self):
        data = {
            "shots": [
                {
                    "shot_id": 0,
                    "role": "hook",
                    "duration_sec": 1.5,
                    "camera": "push_in",
                    "image_index": 0,
                    "transition": "cut",
                }
            ],
            "total_duration_sec": 1.5,
        }
        errors = validate_output("shot_plan", data)
        assert errors == []

    def test_invalid_shot_plan_negative_duration(self):
        data = {
            "shots": [
                {
                    "shot_id": 0,
                    "role": "hook",
                    "duration_sec": -1.0,  # gt=0.0
                }
            ],
            "total_duration_sec": -1.0,  # gt=0.0
        }
        errors = validate_output("shot_plan", data)
        assert len(errors) > 0

    def test_valid_copies(self):
        data = [
            {"hook_line": "테스트 훅 라인", "caption_lines": [], "vo_script": None},
            {"hook_line": None, "caption_lines": [{"text": "테스트", "start_sec": 0, "end_sec": 1, "position": "bottom_center"}]},
        ]
        errors = validate_output("copies", data)
        assert errors == []

    def test_invalid_copies_not_list(self):
        errors = validate_output("copies", {"not": "a list"})
        assert len(errors) > 0
        assert "Expected a JSON array" in errors[0]

    def test_valid_storyboard(self):
        data = {
            "project_id": "test_001",
            "shots": [],
            "total_duration_sec": 0.0,
        }
        errors = validate_output("storyboard", data)
        assert errors == []

    def test_valid_qa_report(self):
        data = {
            "verdict": "PASS",
            "issues": [],
            "revision_count": 0,
        }
        errors = validate_output("qa_report", data)
        assert errors == []

    def test_invalid_qa_report_bad_verdict(self):
        data = {
            "verdict": "MAYBE",
            "issues": [],
        }
        errors = validate_output("qa_report", data)
        assert len(errors) > 0

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown type"):
            validate_output("nonexistent", {})


class TestValidTypes:
    def test_all_expected_types_present(self):
        expected = {"brief", "shot_plan", "copies", "storyboard", "qa_report"}
        assert set(VALID_TYPES) == expected


# ── CLI entry point test ─────────────────────────────────────────────


class TestCLI:
    def test_cli_valid_brief(self, tmp_path: Path):
        data = {
            "concept_keywords": ["로맨틱"],
            "hero_features": ["pool"],
            "narrative_structure": "highlight",
            "target_emotion": "joy",
            "tone_direction": "럭셔리",
            "hook_direction": "훅 스타일",
            "cta_direction": "CTA 스타일",
        }
        path = tmp_path / "brief.json"
        path.write_text(json.dumps(data, ensure_ascii=False))

        result = subprocess.run(
            [sys.executable, "-m", "reels.production.omc_helpers.validate_output", "brief", str(path)],
            capture_output=True,
            text=True,
            cwd="/home/youngwoo/repos/reels-project",
        )
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_cli_invalid_brief(self, tmp_path: Path):
        data = {"concept_keywords": []}
        path = tmp_path / "brief.json"
        path.write_text(json.dumps(data))

        result = subprocess.run(
            [sys.executable, "-m", "reels.production.omc_helpers.validate_output", "brief", str(path)],
            capture_output=True,
            text=True,
            cwd="/home/youngwoo/repos/reels-project",
        )
        assert result.returncode == 1
        assert "FAILED" in result.stderr

    def test_cli_unknown_type(self):
        result = subprocess.run(
            [sys.executable, "-m", "reels.production.omc_helpers.validate_output", "bogus", "x.json"],
            capture_output=True,
            text=True,
            cwd="/home/youngwoo/repos/reels-project",
        )
        assert result.returncode == 1
