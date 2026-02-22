"""Template assembler: combine analysis results into a Template model."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from pydantic import BaseModel

from reels.models.analysis import (
    CameraResult,
    PlaceResult,
    RhythmResult,
    SpeechResult,
    SubtitleResult,
)
from reels.models.metadata import VideoMetadata
from reels.models.shot import Shot
from reels.models.template import (
    AudioInfo,
    BoundingBox,
    CameraInfo,
    CameraType,
    EditInfo,
    Overlay,
    OverlayKind,
    OverlayStyle,
    Template,
    TemplateFormat,
    TemplateMetadata,
    TemplateShot,
    TransitionType,
)

logger = logging.getLogger(__name__)


class TemplateAssembler:
    """Assemble analysis results into a final Template."""

    def __init__(self, config: dict | None = None) -> None:
        syn_cfg = (config or {}).get("synthesis", {})
        self.template_version: str = syn_cfg.get("template_version", "1.0")
        self.normalize_overlay_pos: bool = syn_cfg.get("overlay_position_normalize", True)

    def assemble(
        self,
        shots: list[Shot],
        metadata: VideoMetadata,
        analysis_results: dict[str, list[BaseModel]],
        source_url: str | None = None,
    ) -> Template:
        """Combine shots and analysis results into a Template.

        Args:
            shots: List of segmented shots.
            metadata: Video metadata.
            analysis_results: Dict mapping analyzer name to list of results.
            source_url: Original video URL (optional).

        Returns:
            Complete Template model.
        """
        template_shots = []

        for i, shot in enumerate(shots):
            place = self._get_result(analysis_results, "place", i, PlaceResult())
            camera = self._get_result(analysis_results, "camera", i, CameraResult())
            subtitle = self._get_result(analysis_results, "subtitle", i, SubtitleResult())
            speech = self._get_result(analysis_results, "speech", i, SpeechResult())
            rhythm = self._get_result(analysis_results, "rhythm", i, RhythmResult())

            template_shot = self._build_template_shot(
                shot, place, camera, subtitle, speech, rhythm,
            )
            template_shots.append(template_shot)

        template_id = f"reels_{uuid.uuid4().hex[:8]}"

        # Determine format from metadata
        aspect = self._determine_aspect(metadata.width, metadata.height)
        fmt = TemplateFormat(
            aspect=aspect,
            fps=int(metadata.fps),
            width=metadata.width,
            height=metadata.height,
        )

        tmpl_metadata = TemplateMetadata(
            source_platform=self._guess_platform(source_url),
            original_resolution=metadata.resolution,
        )

        template = Template(
            template_id=template_id,
            source_url=source_url,
            format=fmt,
            total_duration_sec=metadata.duration_sec,
            shot_count=len(template_shots),
            shots=template_shots,
            metadata=tmpl_metadata,
        )

        logger.info("Assembled template %s: %d shots, %.1fs", template_id, len(template_shots), metadata.duration_sec)
        return template

    def _build_template_shot(
        self,
        shot: Shot,
        place: PlaceResult,
        camera: CameraResult,
        subtitle: SubtitleResult,
        speech: SpeechResult,
        rhythm: RhythmResult,
    ) -> TemplateShot:
        """Build a single TemplateShot from analysis results."""
        # Camera info
        camera_type = self._safe_camera_type(camera.camera_type)
        camera_info = CameraInfo(
            type=camera_type,
            shake_score=camera.shake_score,
            speed_factor=1.0,
        )

        # Overlays from subtitles
        overlays = []
        for entry in subtitle.texts:
            box = BoundingBox(
                x=entry.box.get("x", 0.1),
                y=entry.box.get("y", 0.7),
                w=entry.box.get("w", 0.8),
                h=entry.box.get("h", 0.1),
            )
            overlays.append(Overlay(
                kind=OverlayKind.CAPTION,
                start_sec=entry.start_sec,
                end_sec=entry.end_sec,
                box=box,
                style=OverlayStyle(),
                text=entry.text,
            ))

        # Audio info
        speech_text = None
        vo_ref = None
        if speech.has_speech and speech.segments:
            speech_text = " ".join(seg.text for seg in speech.segments)
            vo_ref = f"{{VO_{shot.shot_id + 1}}}"

        audio_info = AudioInfo(
            has_speech=speech.has_speech,
            speech_text=speech_text,
            vo_ref=vo_ref,
            music_cue=rhythm.music_cue,
            beat_aligned=rhythm.beat_aligned,
        )

        # Edit info
        transition = TransitionType.CUT_ON_BEAT if rhythm.beat_aligned else TransitionType.CUT
        edit_info = EditInfo(speed=1.0, transition_out=transition)

        # BPM from rhythm analysis (per-shot)
        shot_bpm = rhythm.bpm if rhythm.bpm > 0 else None

        return TemplateShot(
            shot_id=shot.shot_id,
            start_sec=shot.start_sec,
            end_sec=shot.end_sec,
            duration_sec=shot.duration_sec,
            place_label=place.place_label,
            camera=camera_info,
            keyframe_paths=shot.keyframe_paths,
            overlays=overlays,
            audio=audio_info,
            edit=edit_info,
            bpm=shot_bpm,
        )

    @staticmethod
    def _get_result(
        results: dict[str, list[BaseModel]],
        name: str,
        index: int,
        default: BaseModel,
    ) -> Any:
        """Safely get a result by analyzer name and shot index."""
        analyzer_results = results.get(name, [])
        if index < len(analyzer_results):
            return analyzer_results[index]
        return default

    @staticmethod
    def _safe_camera_type(raw: str) -> CameraType:
        """Convert raw camera type string to CameraType enum."""
        try:
            return CameraType(raw)
        except ValueError:
            return CameraType.STATIC

    @staticmethod
    def _determine_aspect(width: int, height: int) -> str:
        """Determine aspect ratio string from dimensions."""
        ratio = width / height if height > 0 else 1.0
        if ratio < 0.7:
            return "9:16"
        elif ratio > 1.4:
            return "16:9"
        elif 0.9 < ratio < 1.1:
            return "1:1"
        return f"{width}:{height}"

    @staticmethod
    def _guess_platform(url: str | None) -> str | None:
        """Guess source platform from URL."""
        if not url:
            return None
        url_lower = url.lower()
        if "instagram" in url_lower:
            return "instagram"
        if "youtube" in url_lower or "youtu.be" in url_lower:
            return "youtube"
        if "tiktok" in url_lower:
            return "tiktok"
        return None
