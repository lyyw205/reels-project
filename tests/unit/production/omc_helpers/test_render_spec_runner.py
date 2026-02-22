"""Tests for reels.production.omc_helpers.render_spec_runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from reels.production.models import (
    CaptionLine,
    ClaimLevel,
    RenderSpec,
    ShotCopy,
    ShotRole,
    Storyboard,
    StoryboardShot,
)
from reels.production.omc_helpers.render_spec_runner import (
    run_render_spec,
    save_render_outputs,
)


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_shot(
    shot_id: int,
    role: str = "feature",
    duration: float = 2.0,
    hook_line: str | None = None,
    captions: list[str] | None = None,
    vo_script: str | None = None,
    asset_path: str = "img.jpg",
) -> StoryboardShot:
    caption_lines = [CaptionLine(text=t) for t in (captions or [])]
    return StoryboardShot(
        shot_id=shot_id,
        role=ShotRole(role),
        start_sec=shot_id * duration,
        end_sec=(shot_id + 1) * duration,
        duration_sec=duration,
        asset_path=asset_path,
        shot_copy=ShotCopy(
            hook_line=hook_line,
            caption_lines=caption_lines,
            vo_script=vo_script,
        ),
    )


def _make_storyboard(shots: list[StoryboardShot] | None = None) -> Storyboard:
    if shots is None:
        shots = [
            _make_shot(0, role="hook", duration=1.5, hook_line="특별한 숙소"),
            _make_shot(1, role="feature", duration=2.0, captions=["프라이빗 공간", "힐링 시간"], vo_script="프라이빗한 공간입니다."),
            _make_shot(2, role="feature", duration=2.0, captions=["오션뷰"]),
            _make_shot(3, role="support", duration=1.5, captions=["편안한 휴식"]),
            _make_shot(4, role="cta", duration=1.5, captions=["지금 예약하기"]),
        ]
    total = sum(s.duration_sec for s in shots)
    return Storyboard(
        project_id="test_render",
        accommodation_name="테스트숙소",
        total_duration_sec=total,
        shots=shots,
    )


# ── run_render_spec tests ─────────────────────────────────────────────


class TestRunRenderSpec:
    def test_returns_render_spec(self):
        sb = _make_storyboard()
        result = run_render_spec(sb)
        assert isinstance(result, RenderSpec)
        assert result.project_id == "test_render"

    def test_asset_mappings_count(self):
        sb = _make_storyboard()
        result = run_render_spec(sb)
        assert len(result.assets) == len(sb.shots)

    def test_srt_generated(self):
        sb = _make_storyboard()
        result = run_render_spec(sb)
        assert result.captions_srt != ""
        assert "-->" in result.captions_srt

    def test_vo_script_collected(self):
        sb = _make_storyboard()
        result = run_render_spec(sb)
        assert result.vo_script is not None
        assert "프라이빗" in result.vo_script

    def test_empty_storyboard(self):
        sb = _make_storyboard(shots=[])
        result = run_render_spec(sb)
        assert len(result.assets) == 0
        assert result.captions_srt == ""
        assert result.vo_script is None

    def test_no_vo_returns_none(self):
        shots = [
            _make_shot(0, role="hook", hook_line="테스트"),
            _make_shot(1, role="cta", captions=["예약하기"]),
        ]
        sb = _make_storyboard(shots=shots)
        result = run_render_spec(sb)
        assert result.vo_script is None

    def test_config_affects_output(self):
        sb = _make_storyboard()
        config = {
            "production": {
                "render": {
                    "format": {"width": 720, "height": 1280, "fps": 24},
                    "ken_burns_default": False,
                }
            }
        }
        result = run_render_spec(sb, config)
        assert result.format.width == 720
        assert result.format.height == 1280
        assert result.format.fps == 24
        # ken_burns should be False for all assets
        for asset in result.assets:
            assert asset.transform.ken_burns is False


# ── save_render_outputs tests ─────────────────────────────────────────


class TestSaveRenderOutputs:
    def test_saves_all_files(self, tmp_path: Path):
        sb = _make_storyboard()
        spec = run_render_spec(sb)
        written = save_render_outputs(tmp_path, spec)

        assert "render_spec" in written
        assert written["render_spec"].exists()

        assert "captions_srt" in written
        assert written["captions_srt"].exists()
        assert "-->" in written["captions_srt"].read_text()

        assert "vo_script" in written
        assert written["vo_script"].exists()

    def test_skips_srt_when_empty(self, tmp_path: Path):
        sb = _make_storyboard(shots=[])
        spec = run_render_spec(sb)
        written = save_render_outputs(tmp_path, spec)

        assert "render_spec" in written
        assert "captions_srt" not in written
        assert "vo_script" not in written

    def test_creates_output_dir(self, tmp_path: Path):
        nested = tmp_path / "deep" / "nested"
        sb = _make_storyboard()
        spec = run_render_spec(sb)
        written = save_render_outputs(nested, spec)
        assert nested.exists()
        assert written["render_spec"].exists()

    def test_render_spec_json_valid(self, tmp_path: Path):
        sb = _make_storyboard()
        spec = run_render_spec(sb)
        save_render_outputs(tmp_path, spec)

        # Re-parse to verify valid JSON
        data = json.loads((tmp_path / "render_spec.json").read_text())
        assert data["project_id"] == "test_render"
        assert len(data["assets"]) == len(sb.shots)


# ── CLI entry point test ─────────────────────────────────────────────


class TestCLI:
    def test_cli_generates_outputs(self, tmp_path: Path):
        sb = _make_storyboard()
        sb_path = tmp_path / "storyboard.json"
        sb_path.write_text(sb.model_dump_json(indent=2))

        out_dir = tmp_path / "output"

        result = subprocess.run(
            [
                sys.executable, "-m", "reels.production.omc_helpers.render_spec_runner",
                "--storyboard", str(sb_path),
                "--output-dir", str(out_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[4]),
        )
        assert result.returncode == 0
        assert (out_dir / "render_spec.json").exists()
        assert (out_dir / "captions.srt").exists()
        assert "render_spec" in result.stdout

    def test_cli_missing_storyboard(self, tmp_path: Path):
        result = subprocess.run(
            [
                sys.executable, "-m", "reels.production.omc_helpers.render_spec_runner",
                "--storyboard", str(tmp_path / "nonexistent.json"),
                "--output-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[4]),
        )
        assert result.returncode == 1
