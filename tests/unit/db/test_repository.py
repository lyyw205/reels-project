"""Tests for reels.db.repository module."""

from __future__ import annotations

from pathlib import Path

import pytest

from reels.db.repository import TemplateRepository
from reels.models.template import (
    AudioInfo,
    CameraInfo,
    CameraType,
    Template,
    TemplateShot,
)


def _make_template(
    template_id: str = "test_001",
    place: str = "bedroom",
    camera: str = "pan_right",
    has_speech: bool = False,
    duration: float = 10.0,
) -> Template:
    return Template(
        template_id=template_id,
        total_duration_sec=duration,
        shot_count=1,
        shots=[
            TemplateShot(
                shot_id=0,
                start_sec=0.0,
                end_sec=duration,
                duration_sec=duration,
                place_label=place,
                camera=CameraInfo(type=CameraType(camera)),
                audio=AudioInfo(has_speech=has_speech),
            ),
        ],
    )


class TestTemplateRepository:
    def test_save_and_get(self, tmp_path: Path) -> None:
        repo = TemplateRepository(tmp_path / "test.db")
        t = _make_template()
        repo.save(t)

        retrieved = repo.get("test_001")
        assert retrieved is not None
        assert retrieved.template_id == "test_001"
        assert retrieved.shots[0].place_label == "bedroom"
        repo.close()

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        repo = TemplateRepository(tmp_path / "test.db")
        assert repo.get("nonexistent") is None
        repo.close()

    def test_list_all(self, tmp_path: Path) -> None:
        repo = TemplateRepository(tmp_path / "test.db")
        repo.save(_make_template("t1"))
        repo.save(_make_template("t2"))
        repo.save(_make_template("t3"))

        items = repo.list_all()
        assert len(items) == 3
        assert all("template_id" in item for item in items)
        repo.close()

    def test_list_all_with_limit(self, tmp_path: Path) -> None:
        repo = TemplateRepository(tmp_path / "test.db")
        for i in range(5):
            repo.save(_make_template(f"t{i}"))

        items = repo.list_all(limit=2)
        assert len(items) == 2
        repo.close()

    def test_search_by_place(self, tmp_path: Path) -> None:
        repo = TemplateRepository(tmp_path / "test.db")
        repo.save(_make_template("t1", place="bedroom"))
        repo.save(_make_template("t2", place="bathroom"))
        repo.save(_make_template("t3", place="bedroom"))

        results = repo.search_by_place("bedroom")
        assert len(results) == 2
        assert all(r.shots[0].place_label == "bedroom" for r in results)
        repo.close()

    def test_search_by_camera_type(self, tmp_path: Path) -> None:
        repo = TemplateRepository(tmp_path / "test.db")
        repo.save(_make_template("t1", camera="pan_right"))
        repo.save(_make_template("t2", camera="static"))
        repo.save(_make_template("t3", camera="pan_right"))

        results = repo.search_by_camera_type("pan_right")
        assert len(results) == 2
        repo.close()

    def test_search_by_duration(self, tmp_path: Path) -> None:
        repo = TemplateRepository(tmp_path / "test.db")
        repo.save(_make_template("t1", duration=5.0))
        repo.save(_make_template("t2", duration=15.0))
        repo.save(_make_template("t3", duration=30.0))

        results = repo.search_by_duration(min_sec=10, max_sec=20)
        assert len(results) == 1
        assert results[0].total_duration_sec == 15.0
        repo.close()

    def test_delete(self, tmp_path: Path) -> None:
        repo = TemplateRepository(tmp_path / "test.db")
        repo.save(_make_template("t1"))
        assert repo.count() == 1

        deleted = repo.delete("t1")
        assert deleted is True
        assert repo.count() == 0
        assert repo.get("t1") is None
        repo.close()

    def test_delete_nonexistent(self, tmp_path: Path) -> None:
        repo = TemplateRepository(tmp_path / "test.db")
        assert repo.delete("nonexistent") is False
        repo.close()

    def test_count(self, tmp_path: Path) -> None:
        repo = TemplateRepository(tmp_path / "test.db")
        assert repo.count() == 0
        repo.save(_make_template("t1"))
        assert repo.count() == 1
        repo.save(_make_template("t2"))
        assert repo.count() == 2
        repo.close()

    def test_upsert_on_save(self, tmp_path: Path) -> None:
        repo = TemplateRepository(tmp_path / "test.db")
        repo.save(_make_template("t1", place="bedroom"))
        repo.save(_make_template("t1", place="bathroom"))  # Same ID, different place

        assert repo.count() == 1
        t = repo.get("t1")
        assert t is not None
        assert t.shots[0].place_label == "bathroom"
        repo.close()

    def test_avg_bpm_extracted(self, tmp_path: Path) -> None:
        t = _make_template("t_bpm")
        t.shots[0].bpm = 128.0
        repo = TemplateRepository(tmp_path / "test.db")
        repo.save(t)

        items = repo.list_all()
        assert len(items) == 1
        assert items[0]["avg_bpm"] == 128.0
        repo.close()

    def test_avg_bpm_none_when_no_bpm(self, tmp_path: Path) -> None:
        repo = TemplateRepository(tmp_path / "test.db")
        repo.save(_make_template("t_no_bpm"))

        items = repo.list_all()
        assert items[0]["avg_bpm"] is None
        repo.close()


class TestSearchComposite:
    def test_search_single_place(self, tmp_path: Path) -> None:
        repo = TemplateRepository(tmp_path / "test.db")
        repo.save(_make_template("t1", place="bedroom"))
        repo.save(_make_template("t2", place="bathroom"))
        repo.save(_make_template("t3", place="bedroom"))

        results = repo.search_composite(places=["bedroom"])
        assert len(results) == 2
        assert all(r.shots[0].place_label == "bedroom" for r in results)
        repo.close()

    def test_search_duration_range(self, tmp_path: Path) -> None:
        repo = TemplateRepository(tmp_path / "test.db")
        repo.save(_make_template("t1", duration=5.0))
        repo.save(_make_template("t2", duration=15.0))
        repo.save(_make_template("t3", duration=30.0))

        results = repo.search_composite(duration_range=(10.0, 20.0))
        assert len(results) == 1
        assert results[0].total_duration_sec == 15.0
        repo.close()

    def test_search_multiple_filters(self, tmp_path: Path) -> None:
        repo = TemplateRepository(tmp_path / "test.db")
        repo.save(_make_template("t1", place="bedroom", duration=10.0, has_speech=False))
        repo.save(_make_template("t2", place="bedroom", duration=10.0, has_speech=True))
        repo.save(_make_template("t3", place="bathroom", duration=10.0, has_speech=False))

        results = repo.search_composite(places=["bedroom"], has_speech=False)
        assert len(results) == 1
        assert results[0].template_id == "t1"
        repo.close()

    def test_search_camera_types(self, tmp_path: Path) -> None:
        repo = TemplateRepository(tmp_path / "test.db")
        repo.save(_make_template("t1", camera="pan_right"))
        repo.save(_make_template("t2", camera="static"))
        repo.save(_make_template("t3", camera="pan_right"))

        results = repo.search_composite(camera_types=["pan_right"])
        assert len(results) == 2
        repo.close()

    def test_search_empty_results(self, tmp_path: Path) -> None:
        repo = TemplateRepository(tmp_path / "test.db")
        repo.save(_make_template("t1", place="bedroom"))

        results = repo.search_composite(places=["rooftop"])
        assert results == []
        repo.close()

    def test_search_with_limit(self, tmp_path: Path) -> None:
        repo = TemplateRepository(tmp_path / "test.db")
        for i in range(5):
            repo.save(_make_template(f"t{i}", place="bedroom"))

        results = repo.search_composite(places=["bedroom"], limit=3)
        assert len(results) == 3
        repo.close()
