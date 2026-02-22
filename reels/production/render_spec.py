"""RenderSpecGenerator: storyboard → Remotion render spec."""

from __future__ import annotations

from typing import Any

from reels.models.template import TemplateFormat
from reels.production.models import (
    AssetMapping,
    AssetTransform,
    MusicSpec,
    RenderSpec,
    Storyboard,
    StoryboardShot,
)


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp HH:MM:SS,mmm."""
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class RenderSpecGenerator:
    """스토리보드 → Remotion 렌더 스펙 변환."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("production", {}).get("render", {})
        fmt = cfg.get("format", {})
        self.aspect: str = fmt.get("aspect", "9:16")
        self.fps: int = int(fmt.get("fps", 30))
        self.width: int = int(fmt.get("width", 1080))
        self.height: int = int(fmt.get("height", 1920))
        self.ken_burns_default: bool = bool(cfg.get("ken_burns_default", True))
        self.safe_area_margin: float = float(cfg.get("safe_area_margin", 0.15))
        self.default_music_volume: float = float(cfg.get("default_music_volume", 0.3))

    def generate(self, storyboard: Storyboard) -> RenderSpec:
        """Create full render spec from storyboard."""
        assets = self._build_asset_mappings(storyboard)
        captions_srt = self._generate_srt(storyboard)
        vo_script = self._collect_vo_script(storyboard)
        music = self._select_music(storyboard)
        fmt = TemplateFormat(
            aspect=self.aspect,
            fps=self.fps,
            width=self.width,
            height=self.height,
        )
        return RenderSpec(
            project_id=storyboard.project_id,
            format=fmt,
            storyboard=storyboard,
            assets=assets,
            captions_srt=captions_srt,
            vo_script=vo_script,
            music=music,
        )

    def _build_asset_mappings(self, storyboard: Storyboard) -> list[AssetMapping]:
        """Create asset mapping for each shot with transforms."""
        mappings: list[AssetMapping] = []
        for shot in storyboard.shots:
            apply_ken_burns = (
                self.ken_burns_default and shot.asset_type == "image"
            )
            transform = AssetTransform(
                resize_to=(self.width, self.height),
                crop_mode="center",
                speed=shot.edit.speed,
                ken_burns=apply_ken_burns,
            )
            mappings.append(
                AssetMapping(
                    shot_id=shot.shot_id,
                    source_path=shot.asset_path,
                    transform=transform,
                )
            )
        return mappings

    def _generate_srt(self, storyboard: Storyboard) -> str:
        """Generate SRT subtitle string from storyboard captions."""
        entries: list[tuple[float, float, str]] = []

        for shot in storyboard.shots:
            # Include hook_line for shot 0
            if shot.shot_id == 0 and shot.copy.hook_line:
                entries.append((shot.start_sec, shot.end_sec, shot.copy.hook_line))

            for caption in shot.copy.caption_lines:
                # Use caption timing if provided, else fall back to shot timing
                start = caption.start_sec if caption.start_sec or caption.end_sec else shot.start_sec
                end = caption.end_sec if caption.start_sec or caption.end_sec else shot.end_sec
                entries.append((start, end, caption.text))

        if not entries:
            return ""

        lines: list[str] = []
        for idx, (start, end, text) in enumerate(entries, start=1):
            lines.append(str(idx))
            lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
            lines.append(text)
            lines.append("")

        return "\n".join(lines)

    def _collect_vo_script(self, storyboard: Storyboard) -> str | None:
        """Collect VO script from all shots."""
        parts = [
            shot.copy.vo_script
            for shot in storyboard.shots
            if shot.copy.vo_script is not None
        ]
        if not parts:
            return None
        return " ".join(parts)

    def _select_music(self, storyboard: Storyboard) -> MusicSpec:
        """Create music spec based on storyboard metadata."""
        return MusicSpec(
            source="auto",
            volume=self.default_music_volume,
            fade_in_sec=0.5,
            fade_out_sec=1.0,
            bpm_target=None,
        )
