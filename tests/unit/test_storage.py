"""Tests for reels.storage module."""

from __future__ import annotations

from pathlib import Path

import pytest

from reels.models.template import (
    AudioInfo,
    CameraInfo,
    CameraType,
    Template,
    TemplateShot,
)
from reels.storage import TemplateArchiver


def _make_template(template_id: str = "test_001") -> Template:
    return Template(
        template_id=template_id,
        total_duration_sec=10.0,
        shot_count=1,
        shots=[
            TemplateShot(
                shot_id=0,
                start_sec=0.0,
                end_sec=10.0,
                duration_sec=10.0,
                place_label="bedroom",
                camera=CameraInfo(type=CameraType.STATIC),
                audio=AudioInfo(has_speech=False),
                keyframe_paths=["shot_0000_kf_0.jpg"],
            ),
        ],
    )


class TestTemplateArchiver:
    def test_archive_creates_directory(self, tmp_path: Path) -> None:
        archiver = TemplateArchiver(tmp_path / "templates")
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        template = _make_template()
        archive_path = archiver.archive(template, work_dir)

        assert archive_path.exists()
        assert (archive_path / "template.json").exists()

    def test_archive_copies_keyframes(self, tmp_path: Path) -> None:
        archiver = TemplateArchiver(tmp_path / "templates")
        work_dir = tmp_path / "work"
        kf_dir = work_dir / "keyframes"
        kf_dir.mkdir(parents=True)
        (kf_dir / "shot_0000_kf_0.jpg").write_bytes(b"fake image data")
        (kf_dir / "shot_0000_kf_1.jpg").write_bytes(b"fake image data 2")

        template = _make_template()
        archive_path = archiver.archive(template, work_dir)

        archived_kf = archive_path / "keyframes"
        assert archived_kf.exists()
        assert (archived_kf / "shot_0000_kf_0.jpg").exists()
        assert (archived_kf / "shot_0000_kf_1.jpg").exists()

    def test_archive_saves_source_reference(self, tmp_path: Path) -> None:
        archiver = TemplateArchiver(tmp_path / "templates")
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        template = _make_template()
        archiver.archive(template, work_dir, source_video="data/samples/test.mp4")

        source_file = tmp_path / "templates" / "test_001" / "source.txt"
        assert source_file.exists()
        assert source_file.read_text() == "data/samples/test.mp4"

    def test_get_existing(self, tmp_path: Path) -> None:
        archiver = TemplateArchiver(tmp_path / "templates")
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        archiver.archive(_make_template(), work_dir)

        result = archiver.get("test_001")
        assert result is not None
        assert result.name == "test_001"

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        archiver = TemplateArchiver(tmp_path / "templates")
        assert archiver.get("nonexistent") is None

    def test_get_template(self, tmp_path: Path) -> None:
        archiver = TemplateArchiver(tmp_path / "templates")
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        archiver.archive(_make_template(), work_dir)

        template = archiver.get_template("test_001")
        assert template is not None
        assert template.template_id == "test_001"
        assert template.shots[0].place_label == "bedroom"

    def test_get_template_nonexistent(self, tmp_path: Path) -> None:
        archiver = TemplateArchiver(tmp_path / "templates")
        assert archiver.get_template("nonexistent") is None

    def test_get_keyframe_dir(self, tmp_path: Path) -> None:
        archiver = TemplateArchiver(tmp_path / "templates")
        work_dir = tmp_path / "work"
        kf_dir = work_dir / "keyframes"
        kf_dir.mkdir(parents=True)
        (kf_dir / "test.jpg").write_bytes(b"data")

        archiver.archive(_make_template(), work_dir)

        kf_path = archiver.get_keyframe_dir("test_001")
        assert kf_path is not None
        assert (kf_path / "test.jpg").exists()

    def test_list_templates_empty(self, tmp_path: Path) -> None:
        archiver = TemplateArchiver(tmp_path / "templates")
        assert archiver.list_templates() == []

    def test_list_templates(self, tmp_path: Path) -> None:
        archiver = TemplateArchiver(tmp_path / "templates")
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        archiver.archive(_make_template("t1"), work_dir)
        archiver.archive(_make_template("t2"), work_dir)
        archiver.archive(_make_template("t3"), work_dir)

        ids = archiver.list_templates()
        assert ids == ["t1", "t2", "t3"]

    def test_delete(self, tmp_path: Path) -> None:
        archiver = TemplateArchiver(tmp_path / "templates")
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        archiver.archive(_make_template(), work_dir)

        deleted = archiver.delete("test_001")
        assert deleted is True
        assert archiver.get("test_001") is None

    def test_delete_nonexistent(self, tmp_path: Path) -> None:
        archiver = TemplateArchiver(tmp_path / "templates")
        assert archiver.delete("nonexistent") is False

    def test_archive_overwrites_existing(self, tmp_path: Path) -> None:
        archiver = TemplateArchiver(tmp_path / "templates")
        work_dir = tmp_path / "work"
        kf_dir = work_dir / "keyframes"
        kf_dir.mkdir(parents=True)
        (kf_dir / "shot.jpg").write_bytes(b"v1")

        archiver.archive(_make_template(), work_dir)

        # Overwrite with new keyframes
        (kf_dir / "shot.jpg").write_bytes(b"v2")
        archiver.archive(_make_template(), work_dir)

        archived = (tmp_path / "templates" / "test_001" / "keyframes" / "shot.jpg").read_bytes()
        assert archived == b"v2"
