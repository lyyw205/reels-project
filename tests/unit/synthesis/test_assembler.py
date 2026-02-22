"""Tests for reels.synthesis.assembler module."""

from __future__ import annotations

import pytest

from reels.models.analysis import (
    CameraResult,
    PlaceResult,
    RhythmResult,
    SpeechResult,
    SpeechSegment,
    SubtitleEntry,
    SubtitleResult,
)
from reels.models.metadata import VideoMetadata
from reels.models.shot import Shot
from reels.models.template import CameraType, OverlayKind, Template, TransitionType
from reels.synthesis.assembler import TemplateAssembler


MOCK_META = VideoMetadata(
    source="test.mp4", duration_sec=10.0, fps=30.0,
    width=1080, height=1920, resolution="1080x1920",
)

MOCK_SHOTS = [
    Shot(shot_id=0, start_sec=0.0, end_sec=5.0, start_frame=0, end_frame=150, duration_sec=5.0),
    Shot(shot_id=1, start_sec=5.0, end_sec=10.0, start_frame=150, end_frame=300, duration_sec=5.0),
]


class TestTemplateAssembler:
    def test_assemble_basic(self) -> None:
        assembler = TemplateAssembler()
        results = {
            "place": [PlaceResult(place_label="bedroom", confidence=0.9), PlaceResult(place_label="bathroom", confidence=0.8)],
            "camera": [CameraResult(camera_type="pan_right"), CameraResult(camera_type="static")],
            "subtitle": [SubtitleResult(), SubtitleResult()],
            "speech": [SpeechResult(), SpeechResult()],
            "rhythm": [RhythmResult(), RhythmResult()],
        }

        template = assembler.assemble(MOCK_SHOTS, MOCK_META, results)

        assert isinstance(template, Template)
        assert template.shot_count == 2
        assert len(template.shots) == 2
        assert template.shots[0].place_label == "bedroom"
        assert template.shots[0].camera.type == CameraType.PAN_RIGHT
        assert template.shots[1].place_label == "bathroom"
        assert template.format.aspect == "9:16"

    def test_assemble_with_speech(self) -> None:
        assembler = TemplateAssembler()
        results = {
            "place": [PlaceResult()],
            "camera": [CameraResult()],
            "subtitle": [SubtitleResult()],
            "speech": [SpeechResult(has_speech=True, segments=[SpeechSegment(text="Hello", start=0.0, end=1.0)])],
            "rhythm": [RhythmResult()],
        }

        template = assembler.assemble([MOCK_SHOTS[0]], MOCK_META, results)
        assert template.shots[0].audio.has_speech is True
        assert template.shots[0].audio.speech_text == "Hello"
        assert template.shots[0].audio.vo_ref == "{VO_1}"

    def test_assemble_with_subtitles(self) -> None:
        assembler = TemplateAssembler()
        results = {
            "place": [PlaceResult()],
            "camera": [CameraResult()],
            "subtitle": [SubtitleResult(texts=[
                SubtitleEntry(text="Test Caption", box={"x": 0.1, "y": 0.7, "w": 0.8, "h": 0.1}, confidence=0.9, start_sec=0.0, end_sec=2.0),
            ])],
            "speech": [SpeechResult()],
            "rhythm": [RhythmResult()],
        }

        template = assembler.assemble([MOCK_SHOTS[0]], MOCK_META, results)
        assert len(template.shots[0].overlays) == 1
        assert template.shots[0].overlays[0].kind == OverlayKind.CAPTION
        assert template.shots[0].overlays[0].text == "Test Caption"

    def test_assemble_beat_aligned_transition(self) -> None:
        assembler = TemplateAssembler()
        results = {
            "place": [PlaceResult()],
            "camera": [CameraResult()],
            "subtitle": [SubtitleResult()],
            "speech": [SpeechResult()],
            "rhythm": [RhythmResult(beat_aligned=True, music_cue="beat")],
        }

        template = assembler.assemble([MOCK_SHOTS[0]], MOCK_META, results)
        assert template.shots[0].edit.transition_out == TransitionType.CUT_ON_BEAT

    def test_assemble_missing_results_uses_defaults(self) -> None:
        assembler = TemplateAssembler()
        # Empty analysis results
        template = assembler.assemble(MOCK_SHOTS, MOCK_META, {})

        assert template.shot_count == 2
        assert template.shots[0].place_label == "other"
        assert template.shots[0].camera.type == CameraType.STATIC

    def test_determine_aspect(self) -> None:
        assert TemplateAssembler._determine_aspect(1080, 1920) == "9:16"
        assert TemplateAssembler._determine_aspect(1920, 1080) == "16:9"
        assert TemplateAssembler._determine_aspect(1080, 1080) == "1:1"

    def test_guess_platform(self) -> None:
        assert TemplateAssembler._guess_platform("https://www.instagram.com/reel/abc") == "instagram"
        assert TemplateAssembler._guess_platform("https://youtube.com/shorts/xyz") == "youtube"
        assert TemplateAssembler._guess_platform("https://tiktok.com/@user/video/123") == "tiktok"
        assert TemplateAssembler._guess_platform(None) is None

    def test_safe_camera_type_valid(self) -> None:
        assert TemplateAssembler._safe_camera_type("pan_right") == CameraType.PAN_RIGHT

    def test_safe_camera_type_invalid(self) -> None:
        assert TemplateAssembler._safe_camera_type("unknown_motion") == CameraType.STATIC

    def test_source_url_passed_through(self) -> None:
        assembler = TemplateAssembler()
        template = assembler.assemble(MOCK_SHOTS, MOCK_META, {}, source_url="https://instagram.com/reel/test")
        assert template.source_url == "https://instagram.com/reel/test"
        assert template.metadata.source_platform == "instagram"

    def test_bpm_wired_from_rhythm(self) -> None:
        assembler = TemplateAssembler()
        results = {
            "place": [PlaceResult()],
            "camera": [CameraResult()],
            "subtitle": [SubtitleResult()],
            "speech": [SpeechResult()],
            "rhythm": [RhythmResult(bpm=120.5, beat_aligned=True)],
        }
        template = assembler.assemble([MOCK_SHOTS[0]], MOCK_META, results)
        assert template.shots[0].bpm == 120.5

    def test_bpm_none_when_zero(self) -> None:
        assembler = TemplateAssembler()
        results = {
            "place": [PlaceResult()],
            "camera": [CameraResult()],
            "subtitle": [SubtitleResult()],
            "speech": [SpeechResult()],
            "rhythm": [RhythmResult(bpm=0.0)],
        }
        template = assembler.assemble([MOCK_SHOTS[0]], MOCK_META, results)
        assert template.shots[0].bpm is None
