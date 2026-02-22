"""Tests for reels.production.render_spec.RenderSpecGenerator."""

from __future__ import annotations

import pytest

from reels.production.models import (
    CaptionLine,
    MusicSpec,
    RenderSpec,
    ShotCopy,
    ShotRole,
    Storyboard,
    StoryboardShot,
)
from reels.production.render_spec import RenderSpecGenerator, _format_srt_time
from reels.models.template import EditInfo


# ─── Helpers ─────────────────────────────────────────────────────


def _make_shot(
    shot_id: int = 0,
    start_sec: float = 0.0,
    end_sec: float = 2.0,
    asset_type: str = "image",
    asset_path: str = "/tmp/img.jpg",
    role: ShotRole = ShotRole.FEATURE,
    hook_line: str | None = None,
    caption_lines: list[CaptionLine] | None = None,
    vo_script: str | None = None,
    speed: float = 1.0,
) -> StoryboardShot:
    copy = ShotCopy(
        hook_line=hook_line,
        caption_lines=caption_lines or [],
        vo_script=vo_script,
    )
    edit = EditInfo(speed=speed)
    return StoryboardShot(
        shot_id=shot_id,
        role=role,
        start_sec=start_sec,
        end_sec=end_sec,
        duration_sec=end_sec - start_sec,
        asset_type=asset_type,
        asset_path=asset_path,
        copy=copy,
        edit=edit,
    )


def _make_storyboard(
    shots: list[StoryboardShot] | None = None,
    project_id: str = "proj-1",
    template_ref: str | None = None,
) -> Storyboard:
    return Storyboard(
        project_id=project_id,
        shots=shots or [],
        template_ref=template_ref,
    )


def _make_generator(config: dict | None = None) -> RenderSpecGenerator:
    return RenderSpecGenerator(config=config)


# ─── generate() ──────────────────────────────────────────────────


class TestGenerate:
    def test_generate_returns_render_spec(self) -> None:
        """generate() returns a RenderSpec instance."""
        gen = _make_generator()
        sb = _make_storyboard(shots=[_make_shot(shot_id=0)])
        result = gen.generate(sb)
        assert isinstance(result, RenderSpec)

    def test_generate_project_id(self) -> None:
        """RenderSpec.project_id matches storyboard.project_id."""
        gen = _make_generator()
        sb = _make_storyboard(project_id="my-project")
        result = gen.generate(sb)
        assert result.project_id == "my-project"

    def test_generate_all_fields_populated(self) -> None:
        """generate() populates assets, captions_srt, and music."""
        gen = _make_generator()
        shots = [
            _make_shot(shot_id=0, hook_line="최고의 뷰", vo_script="환상적인 전망"),
            _make_shot(
                shot_id=1,
                start_sec=2.0,
                end_sec=4.0,
                caption_lines=[CaptionLine(text="노천탕", start_sec=2.0, end_sec=4.0)],
                vo_script="편안한 휴식",
            ),
        ]
        sb = _make_storyboard(shots=shots)
        result = gen.generate(sb)

        assert len(result.assets) == 2
        assert result.captions_srt != ""
        assert result.vo_script is not None
        assert result.music is not None

    def test_generate_format_defaults(self) -> None:
        """Default format is 9:16, 1080x1920, 30fps."""
        gen = _make_generator()
        sb = _make_storyboard()
        result = gen.generate(sb)

        assert result.format.aspect == "9:16"
        assert result.format.fps == 30
        assert result.format.width == 1080
        assert result.format.height == 1920

    def test_generate_storyboard_reference(self) -> None:
        """RenderSpec.storyboard is the same storyboard passed in."""
        gen = _make_generator()
        sb = _make_storyboard()
        result = gen.generate(sb)
        assert result.storyboard is sb


# ─── _build_asset_mappings ───────────────────────────────────────


class TestBuildAssetMappings:
    def test_correct_number_of_mappings(self) -> None:
        """One AssetMapping per shot."""
        gen = _make_generator()
        shots = [_make_shot(shot_id=i) for i in range(5)]
        sb = _make_storyboard(shots=shots)
        mappings = gen._build_asset_mappings(sb)
        assert len(mappings) == 5

    def test_shot_id_preserved(self) -> None:
        """AssetMapping.shot_id matches StoryboardShot.shot_id."""
        gen = _make_generator()
        shots = [_make_shot(shot_id=7)]
        sb = _make_storyboard(shots=shots)
        mappings = gen._build_asset_mappings(sb)
        assert mappings[0].shot_id == 7

    def test_source_path_from_shot(self) -> None:
        """AssetMapping.source_path equals shot.asset_path."""
        gen = _make_generator()
        shots = [_make_shot(asset_path="/data/hotel.jpg")]
        sb = _make_storyboard(shots=shots)
        mappings = gen._build_asset_mappings(sb)
        assert mappings[0].source_path == "/data/hotel.jpg"

    def test_ken_burns_applied_for_image(self) -> None:
        """asset_type='image' → transform.ken_burns=True (default config)."""
        gen = _make_generator()
        shots = [_make_shot(asset_type="image")]
        sb = _make_storyboard(shots=shots)
        mappings = gen._build_asset_mappings(sb)
        assert mappings[0].transform.ken_burns is True

    def test_ken_burns_not_applied_for_video(self) -> None:
        """asset_type='video_clip' → transform.ken_burns=False."""
        gen = _make_generator()
        shots = [_make_shot(asset_type="video_clip", asset_path="/tmp/v.mp4")]
        sb = _make_storyboard(shots=shots)
        mappings = gen._build_asset_mappings(sb)
        assert mappings[0].transform.ken_burns is False

    def test_resize_to_defaults(self) -> None:
        """transform.resize_to is (1080, 1920) by default."""
        gen = _make_generator()
        shots = [_make_shot()]
        sb = _make_storyboard(shots=shots)
        mappings = gen._build_asset_mappings(sb)
        assert mappings[0].transform.resize_to == (1080, 1920)

    def test_crop_mode_center(self) -> None:
        """transform.crop_mode is 'center'."""
        gen = _make_generator()
        shots = [_make_shot()]
        sb = _make_storyboard(shots=shots)
        mappings = gen._build_asset_mappings(sb)
        assert mappings[0].transform.crop_mode == "center"

    def test_speed_from_shot_edit(self) -> None:
        """transform.speed comes from shot.edit.speed."""
        gen = _make_generator()
        shots = [_make_shot(speed=1.5)]
        sb = _make_storyboard(shots=shots)
        mappings = gen._build_asset_mappings(sb)
        assert mappings[0].transform.speed == pytest.approx(1.5)

    def test_ken_burns_disabled_by_config(self) -> None:
        """ken_burns_default=False in config → no ken_burns even for images."""
        config = {"production": {"render": {"ken_burns_default": False}}}
        gen = _make_generator(config=config)
        shots = [_make_shot(asset_type="image")]
        sb = _make_storyboard(shots=shots)
        mappings = gen._build_asset_mappings(sb)
        assert mappings[0].transform.ken_burns is False


# ─── _generate_srt ───────────────────────────────────────────────


class TestGenerateSrt:
    def test_valid_srt_format(self) -> None:
        """SRT output contains index, timestamp line, and text."""
        gen = _make_generator()
        shots = [
            _make_shot(
                shot_id=0,
                start_sec=0.0,
                end_sec=2.0,
                caption_lines=[CaptionLine(text="Hello", start_sec=0.0, end_sec=2.0)],
            )
        ]
        sb = _make_storyboard(shots=shots)
        srt = gen._generate_srt(sb)

        assert "1\n" in srt
        assert "00:00:00,000 --> 00:00:02,000" in srt
        assert "Hello" in srt

    def test_correct_numbering(self) -> None:
        """Multiple captions get sequential SRT numbers."""
        gen = _make_generator()
        shots = [
            _make_shot(
                shot_id=0,
                start_sec=0.0,
                end_sec=2.0,
                caption_lines=[
                    CaptionLine(text="First", start_sec=0.0, end_sec=1.0),
                    CaptionLine(text="Second", start_sec=1.0, end_sec=2.0),
                ],
            )
        ]
        sb = _make_storyboard(shots=shots)
        srt = gen._generate_srt(sb)

        lines = srt.strip().split("\n")
        # entries: 1, timestamp, text, blank repeated
        assert lines[0] == "1"
        assert lines[4] == "2"

    def test_includes_hook_line_for_shot_0(self) -> None:
        """hook_line from shot 0 appears in SRT output."""
        gen = _make_generator()
        shots = [
            _make_shot(shot_id=0, start_sec=0.0, end_sec=2.0, hook_line="눈부신 오션뷰")
        ]
        sb = _make_storyboard(shots=shots)
        srt = gen._generate_srt(sb)

        assert "눈부신 오션뷰" in srt

    def test_hook_line_only_for_shot_0(self) -> None:
        """hook_line from shot 1+ is not included in SRT."""
        gen = _make_generator()
        shots = [
            _make_shot(shot_id=0, start_sec=0.0, end_sec=2.0),
            _make_shot(shot_id=1, start_sec=2.0, end_sec=4.0, hook_line="Should not appear"),
        ]
        sb = _make_storyboard(shots=shots)
        srt = gen._generate_srt(sb)

        assert "Should not appear" not in srt

    def test_empty_captions_returns_empty_string(self) -> None:
        """No captions and no hook → empty SRT string."""
        gen = _make_generator()
        shots = [_make_shot(shot_id=0)]
        sb = _make_storyboard(shots=shots)
        srt = gen._generate_srt(sb)

        assert srt == ""

    def test_no_shots_returns_empty_string(self) -> None:
        """No shots → empty SRT."""
        gen = _make_generator()
        sb = _make_storyboard(shots=[])
        srt = gen._generate_srt(sb)
        assert srt == ""

    def test_caption_without_timing_uses_shot_timing(self) -> None:
        """CaptionLine with start_sec=0.0/end_sec=0.0 falls back to shot timing."""
        gen = _make_generator()
        shots = [
            _make_shot(
                shot_id=0,
                start_sec=3.0,
                end_sec=6.0,
                caption_lines=[CaptionLine(text="Fallback", start_sec=0.0, end_sec=0.0)],
            )
        ]
        sb = _make_storyboard(shots=shots)
        srt = gen._generate_srt(sb)

        assert "00:00:03,000 --> 00:00:06,000" in srt
        assert "Fallback" in srt

    def test_multiple_shots_captions(self) -> None:
        """Captions from multiple shots all appear in SRT with correct indices."""
        gen = _make_generator()
        shots = [
            _make_shot(
                shot_id=0,
                start_sec=0.0,
                end_sec=2.0,
                hook_line="Hook",
            ),
            _make_shot(
                shot_id=1,
                start_sec=2.0,
                end_sec=4.0,
                caption_lines=[CaptionLine(text="Cap1", start_sec=2.0, end_sec=4.0)],
            ),
        ]
        sb = _make_storyboard(shots=shots)
        srt = gen._generate_srt(sb)

        assert "Hook" in srt
        assert "Cap1" in srt
        # Index 1 and 2 both present
        assert "\n1\n" in srt or srt.startswith("1\n")
        assert "2\n" in srt


# ─── _collect_vo_script ──────────────────────────────────────────


class TestCollectVoScript:
    def test_joins_scripts_from_multiple_shots(self) -> None:
        """VO scripts from multiple shots are joined with spaces."""
        gen = _make_generator()
        shots = [
            _make_shot(shot_id=0, vo_script="환상적인"),
            _make_shot(shot_id=1, start_sec=2.0, end_sec=4.0, vo_script="전망입니다"),
        ]
        sb = _make_storyboard(shots=shots)
        result = gen._collect_vo_script(sb)
        assert result == "환상적인 전망입니다"

    def test_returns_none_when_no_vo(self) -> None:
        """All shots have vo_script=None → return None."""
        gen = _make_generator()
        shots = [_make_shot(shot_id=0), _make_shot(shot_id=1, start_sec=2.0, end_sec=4.0)]
        sb = _make_storyboard(shots=shots)
        result = gen._collect_vo_script(sb)
        assert result is None

    def test_skips_none_vo_scripts(self) -> None:
        """Shots without VO are skipped; only non-None values joined."""
        gen = _make_generator()
        shots = [
            _make_shot(shot_id=0, vo_script="첫번째"),
            _make_shot(shot_id=1, start_sec=2.0, end_sec=4.0, vo_script=None),
            _make_shot(shot_id=2, start_sec=4.0, end_sec=6.0, vo_script="세번째"),
        ]
        sb = _make_storyboard(shots=shots)
        result = gen._collect_vo_script(sb)
        assert result == "첫번째 세번째"

    def test_single_shot_vo(self) -> None:
        """Single shot VO returned as-is (no extra spaces)."""
        gen = _make_generator()
        shots = [_make_shot(shot_id=0, vo_script="Only this")]
        sb = _make_storyboard(shots=shots)
        result = gen._collect_vo_script(sb)
        assert result == "Only this"

    def test_no_shots_returns_none(self) -> None:
        """No shots → None."""
        gen = _make_generator()
        sb = _make_storyboard(shots=[])
        result = gen._collect_vo_script(sb)
        assert result is None


# ─── _select_music ───────────────────────────────────────────────


class TestSelectMusic:
    def test_returns_music_spec(self) -> None:
        """_select_music returns a MusicSpec instance."""
        gen = _make_generator()
        sb = _make_storyboard()
        result = gen._select_music(sb)
        assert isinstance(result, MusicSpec)

    def test_default_source_auto(self) -> None:
        """Default music source is 'auto'."""
        gen = _make_generator()
        sb = _make_storyboard()
        result = gen._select_music(sb)
        assert result.source == "auto"

    def test_default_volume(self) -> None:
        """Default volume is 0.3."""
        gen = _make_generator()
        sb = _make_storyboard()
        result = gen._select_music(sb)
        assert result.volume == pytest.approx(0.3)

    def test_default_fade_in(self) -> None:
        """Default fade_in_sec is 0.5."""
        gen = _make_generator()
        sb = _make_storyboard()
        result = gen._select_music(sb)
        assert result.fade_in_sec == pytest.approx(0.5)

    def test_default_fade_out(self) -> None:
        """Default fade_out_sec is 1.0."""
        gen = _make_generator()
        sb = _make_storyboard()
        result = gen._select_music(sb)
        assert result.fade_out_sec == pytest.approx(1.0)

    def test_bpm_target_default_none(self) -> None:
        """Default bpm_target is None."""
        gen = _make_generator()
        sb = _make_storyboard()
        result = gen._select_music(sb)
        assert result.bpm_target is None


# ─── Config overrides ────────────────────────────────────────────


class TestConfigOverrides:
    def test_custom_volume(self) -> None:
        """Custom default_music_volume in config is applied."""
        config = {"production": {"render": {"default_music_volume": 0.5}}}
        gen = _make_generator(config=config)
        sb = _make_storyboard()
        result = gen._select_music(sb)
        assert result.volume == pytest.approx(0.5)

    def test_custom_format(self) -> None:
        """Custom format fields in config are used in generate()."""
        config = {
            "production": {
                "render": {
                    "format": {"aspect": "1:1", "fps": 24, "width": 1080, "height": 1080}
                }
            }
        }
        gen = _make_generator(config=config)
        sb = _make_storyboard()
        result = gen.generate(sb)
        assert result.format.aspect == "1:1"
        assert result.format.fps == 24
        assert result.format.height == 1080

    def test_custom_resize_in_mappings(self) -> None:
        """Custom width/height are reflected in AssetTransform.resize_to."""
        config = {
            "production": {
                "render": {
                    "format": {"width": 720, "height": 1280}
                }
            }
        }
        gen = _make_generator(config=config)
        shots = [_make_shot()]
        sb = _make_storyboard(shots=shots)
        mappings = gen._build_asset_mappings(sb)
        assert mappings[0].transform.resize_to == (720, 1280)

    def test_empty_config_uses_defaults(self) -> None:
        """Passing {} config still uses all defaults."""
        gen = _make_generator(config={})
        assert gen.aspect == "9:16"
        assert gen.fps == 30
        assert gen.width == 1080
        assert gen.height == 1920
        assert gen.ken_burns_default is True
        assert gen.safe_area_margin == pytest.approx(0.15)
        assert gen.default_music_volume == pytest.approx(0.3)

    def test_none_config_uses_defaults(self) -> None:
        """Passing None config uses all defaults."""
        gen = _make_generator(config=None)
        assert gen.fps == 30
        assert gen.ken_burns_default is True


# ─── _format_srt_time helper ─────────────────────────────────────


class TestFormatSrtTime:
    @pytest.mark.parametrize("seconds,expected", [
        (0.0, "00:00:00,000"),
        (1.2, "00:00:01,200"),
        (61.5, "00:01:01,500"),
        (3661.999, "01:01:01,999"),
        (0.001, "00:00:00,001"),
    ])
    def test_format_srt_time(self, seconds: float, expected: str) -> None:
        assert _format_srt_time(seconds) == expected
