"""Integration test for the full pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from reels.models.template import Template


@pytest.mark.slow
class TestFullPipeline:
    def test_pipeline_with_synthetic_video(
        self, synthetic_video: Path | None, tmp_path: Path,
    ) -> None:
        """Run full pipeline on synthetic test video."""
        if synthetic_video is None:
            pytest.skip("ffmpeg not available")

        from reels.pipeline import run_pipeline

        config = {
            "pipeline": {"max_video_duration_sec": 300, "cache_enabled": False},
            "ingest": {
                "normalize": {
                    "strategy": "preserve_aspect",
                    "max_short_side": 320,
                    "target_fps": 30,
                    "audio_sample_rate": 16000,
                    "video_codec": "libx264",
                    "audio_codec": "aac",
                }
            },
            "segmentation": {
                "threshold": 20.0,
                "min_shot_duration_sec": 0.25,
                "merge_similar_threshold": 0.92,
            },
            "analysis": {
                "place": {"keyframes_per_shot": 1},
            },
            "db": {"path": str(tmp_path / "test.db")},
        }

        work_dir = tmp_path / "work"
        template = run_pipeline(
            source=str(synthetic_video),
            work_dir=work_dir,
            config=config,
            save_to_db=True,
        )

        assert isinstance(template, Template)
        assert template.shot_count > 0
        assert template.total_duration_sec > 0

        # Verify template JSON was saved
        template_file = work_dir / f"{template.template_id}.json"
        assert template_file.exists()

        # Verify DB save
        from reels.db.repository import TemplateRepository
        repo = TemplateRepository(tmp_path / "test.db")
        retrieved = repo.get(template.template_id)
        assert retrieved is not None
        repo.close()
