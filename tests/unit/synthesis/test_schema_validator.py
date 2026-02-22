"""Tests for reels.synthesis.schema_validator module."""

from __future__ import annotations

import json

import pytest

from reels.exceptions import SynthesisError
from reels.models.template import Template, TemplateShot
from reels.synthesis.schema_validator import validate_template, validate_template_json


def _make_template(**kwargs) -> Template:
    """Helper to create a minimal valid template."""
    defaults = dict(
        template_id="test_001",
        total_duration_sec=10.0,
        shot_count=2,
        shots=[
            TemplateShot(shot_id=0, start_sec=0.0, end_sec=5.0, duration_sec=5.0),
            TemplateShot(shot_id=1, start_sec=5.0, end_sec=10.0, duration_sec=5.0),
        ],
    )
    defaults.update(kwargs)
    return Template(**defaults)


class TestValidateTemplate:
    def test_valid_template(self) -> None:
        t = _make_template()
        assert validate_template(t) is True

    def test_shot_count_mismatch(self) -> None:
        t = _make_template(shot_count=5)
        with pytest.raises(SynthesisError, match="shot_count"):
            validate_template(t)

    def test_negative_duration(self) -> None:
        shots = [
            TemplateShot(shot_id=0, start_sec=5.0, end_sec=3.0, duration_sec=-2.0),
        ]
        t = _make_template(shot_count=1, shots=shots)
        with pytest.raises(SynthesisError, match="negative duration"):
            validate_template(t)

    def test_end_before_start(self) -> None:
        shots = [
            TemplateShot(shot_id=0, start_sec=5.0, end_sec=2.0, duration_sec=-3.0),
        ]
        t = _make_template(shot_count=1, shots=shots)
        with pytest.raises(SynthesisError):
            validate_template(t)


class TestValidateTemplateJson:
    def test_valid_json(self) -> None:
        t = _make_template()
        json_str = t.model_dump_json()
        result = validate_template_json(json_str)
        assert result.template_id == "test_001"

    def test_invalid_json(self) -> None:
        with pytest.raises(SynthesisError, match="Invalid JSON"):
            validate_template_json("{not valid json")

    def test_schema_error(self) -> None:
        with pytest.raises(SynthesisError, match="schema error"):
            validate_template_json('{"template_id": "x"}')
