"""Tests for reels.models package."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reels.models.metadata import AudioMetadata, IngestResult, NormalizeResult, VideoMetadata
from reels.models.shot import SegmentationResult, Shot, ShotBoundary
from reels.models.analysis import (
    CameraResult,
    PlaceResult,
    RhythmResult,
    SpeechResult,
    SpeechSegment,
    SpeechWord,
    SubtitleEntry,
    SubtitleResult,
)
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

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ─── VideoMetadata ───────────────────────────────────────────────

class TestVideoMetadata:
    def test_basic(self) -> None:
        m = VideoMetadata(source="v.mp4", duration_sec=10.0, fps=30.0, width=1080, height=1920, resolution="1080x1920")
        assert m.source == "v.mp4"
        assert m.has_audio is True
        assert m.codec is None

    def test_from_fixture(self) -> None:
        data = json.loads((FIXTURES_DIR / "sample_metadata.json").read_text())
        m = VideoMetadata(**data)
        assert m.fps == 30.0
        assert m.resolution == "1080x1920"

    def test_negative_duration_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VideoMetadata(source="v.mp4", duration_sec=-1.0, fps=30.0, width=1080, height=1920, resolution="x")

    def test_zero_fps_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VideoMetadata(source="v.mp4", duration_sec=1.0, fps=0, width=1080, height=1920, resolution="x")


class TestAudioMetadata:
    def test_basic(self) -> None:
        a = AudioMetadata(path="a.wav", sample_rate=16000, duration_sec=5.0)
        assert a.channels == 1


# ─── Shot models ─────────────────────────────────────────────────

class TestShotBoundary:
    def test_basic(self) -> None:
        b = ShotBoundary(frame_number=90, timecode_sec=3.0)
        assert b.confidence == 1.0

    def test_confidence_range(self) -> None:
        with pytest.raises(ValidationError):
            ShotBoundary(frame_number=0, timecode_sec=0.0, confidence=1.5)


class TestShot:
    def test_basic(self) -> None:
        s = Shot(shot_id=0, start_sec=0.0, end_sec=3.0, start_frame=0, end_frame=90, duration_sec=3.0)
        assert s.keyframe_paths == []

    def test_from_fixture(self) -> None:
        data = json.loads((FIXTURES_DIR / "sample_shots.json").read_text())
        shots = [Shot(**d) for d in data]
        assert len(shots) == 5
        assert shots[0].start_sec == 0.0
        assert shots[-1].end_sec == 15.0


class TestSegmentationResult:
    def test_auto_count(self) -> None:
        shots = [Shot(shot_id=i, start_sec=float(i), end_sec=float(i + 1), start_frame=i * 30, end_frame=(i + 1) * 30, duration_sec=1.0) for i in range(3)]
        r = SegmentationResult(shots=shots, keyframe_dir="/tmp/kf")
        assert r.total_shots == 3

    def test_explicit_count(self) -> None:
        r = SegmentationResult(shots=[], keyframe_dir="/tmp/kf", total_shots=5)
        assert r.total_shots == 5


# ─── Analysis models ─────────────────────────────────────────────

class TestPlaceResult:
    def test_defaults(self) -> None:
        p = PlaceResult()
        assert p.place_label == "other"
        assert p.confidence == 0.0

    def test_with_labels(self) -> None:
        p = PlaceResult(place_label="bedroom", confidence=0.85, top_labels=[("bedroom", 0.85), ("bathroom", 0.1)])
        assert p.top_labels[0][0] == "bedroom"


class TestCameraResult:
    def test_defaults(self) -> None:
        c = CameraResult()
        assert c.camera_type == "static"
        assert c.shake_score == 0.0


class TestSubtitleEntry:
    def test_basic(self) -> None:
        e = SubtitleEntry(text="Hello", start_sec=0.0, end_sec=1.0, confidence=0.9)
        assert e.text == "Hello"


class TestSpeechResult:
    def test_empty(self) -> None:
        r = SpeechResult()
        assert r.has_speech is False
        assert r.segments == []

    def test_with_words(self) -> None:
        w = SpeechWord(word="hello", start=0.0, end=0.5, probability=0.95)
        seg = SpeechSegment(text="hello world", start=0.0, end=1.0, words=[w])
        r = SpeechResult(has_speech=True, segments=[seg])
        assert r.segments[0].words[0].word == "hello"


class TestRhythmResult:
    def test_defaults(self) -> None:
        r = RhythmResult()
        assert r.bpm == 0.0
        assert r.beat_aligned is False


# ─── Template models ─────────────────────────────────────────────

class TestStrEnums:
    def test_overlay_kind_values(self) -> None:
        assert OverlayKind.CAPTION == "caption"
        assert OverlayKind.CTA == "cta"

    def test_camera_type_values(self) -> None:
        assert CameraType.GIMBAL_SMOOTH == "gimbal_smooth"
        assert CameraType.PAN_RIGHT == "pan_right"

    def test_transition_type_values(self) -> None:
        assert TransitionType.CUT_ON_BEAT == "cut_on_beat"


class TestBoundingBox:
    def test_valid(self) -> None:
        b = BoundingBox(x=0.1, y=0.7, w=0.8, h=0.1)
        assert b.x == 0.1

    def test_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            BoundingBox(x=1.5, y=0.0, w=0.5, h=0.5)


class TestOverlay:
    def test_basic(self) -> None:
        o = Overlay(
            kind=OverlayKind.CAPTION,
            start_sec=0.0,
            end_sec=1.0,
            box=BoundingBox(x=0.1, y=0.7, w=0.8, h=0.1),
            text="{HOOK_LINE}",
        )
        assert o.animation is None
        assert o.style.fill == "#FFFFFF"


class TestTemplateShot:
    def test_defaults(self) -> None:
        ts = TemplateShot(shot_id=0, start_sec=0.0, end_sec=3.0, duration_sec=3.0)
        assert ts.place_label == "other"
        assert ts.camera.type == CameraType.STATIC
        assert ts.edit.transition_out == TransitionType.CUT


class TestTemplate:
    def test_from_fixture(self) -> None:
        data = json.loads((FIXTURES_DIR / "sample_template.json").read_text())
        t = Template(**data)
        assert t.template_id == "reels_test_001"
        assert t.shot_count == 2
        assert len(t.shots) == 2
        assert t.shots[0].overlays[0].kind == OverlayKind.CAPTION
        assert t.shots[0].camera.type == CameraType.GIMBAL_SMOOTH
        assert t.shots[1].edit.transition_out == TransitionType.CUT

    def test_minimal(self) -> None:
        t = Template(
            template_id="t1",
            total_duration_sec=5.0,
            shot_count=1,
            shots=[TemplateShot(shot_id=0, start_sec=0.0, end_sec=5.0, duration_sec=5.0)],
        )
        assert t.format.aspect == "9:16"
        assert t.metadata.source_platform is None

    def test_serialization_roundtrip(self) -> None:
        data = json.loads((FIXTURES_DIR / "sample_template.json").read_text())
        t = Template(**data)
        roundtrip = Template(**json.loads(t.model_dump_json()))
        assert roundtrip.template_id == t.template_id
        assert len(roundtrip.shots) == len(t.shots)
