"""Pipeline orchestrator with state recovery."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reels.analysis import (
    AnalysisContext,
    AnalysisRunner,
    CameraAnalyzer,
    PlaceAnalyzer,
    RhythmAnalyzer,
    SpeechAnalyzer,
    SubtitleAnalyzer,
)
from reels.db.repository import TemplateRepository
from reels.ingest import ingest_video
from reels.models.template import Template
from reels.segmentation import segment_video
from reels.storage import TemplateArchiver
from reels.synthesis import synthesize_template

logger = logging.getLogger(__name__)

STAGES = ["ingest", "segmentation", "analysis", "synthesis"]


class PipelineState:
    """Track pipeline execution state for resume support."""

    def __init__(self, work_dir: Path) -> None:
        self.path = work_dir / "pipeline_state.json"
        self._state: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._state = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                self._state = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._state, indent=2, default=str))

    @property
    def completed_stages(self) -> list[str]:
        return self._state.get("completed_stages", [])

    def mark_complete(self, stage: str, data: dict[str, Any] | None = None) -> None:
        completed = self.completed_stages
        if stage not in completed:
            completed.append(stage)
        self._state["completed_stages"] = completed
        self._state["current_stage"] = None
        if data:
            self._state.setdefault("stage_data", {})[stage] = data
        self._save()

    def mark_started(self, stage: str) -> None:
        self._state["current_stage"] = stage
        self._state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def get_stage_data(self, stage: str) -> dict[str, Any]:
        return self._state.get("stage_data", {}).get(stage, {})

    def should_skip(self, stage: str) -> bool:
        return stage in self.completed_stages

    def reset(self) -> None:
        self._state = {}
        if self.path.exists():
            self.path.unlink()

    def init(self, source: str) -> None:
        self._state = {
            "source": source,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_stages": [],
            "current_stage": None,
            "stage_data": {},
        }
        self._save()


class AnalysisPipeline:
    """Thin wrapper around run_pipeline for object-oriented usage."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def run(
        self,
        source: str,
        work_dir: Path,
        save_to_db: bool = False,
        resume: bool = False,
    ) -> "Template":
        """Run the full analysis pipeline."""
        return run_pipeline(
            source=source,
            work_dir=work_dir,
            config=self.config,
            save_to_db=save_to_db,
            resume=resume,
        )


def run_pipeline(
    source: str,
    work_dir: Path,
    config: dict[str, Any],
    save_to_db: bool = False,
    resume: bool = False,
) -> Template:
    """Run the full analysis pipeline.

    Stages: ingest -> segmentation -> analysis -> synthesis.
    Each stage saves state to pipeline_state.json for resume support.

    Args:
        source: URL or local file path.
        work_dir: Working directory for intermediate files.
        config: Pipeline configuration.
        save_to_db: Save final template to SQLite DB.
        resume: Resume from last completed stage.

    Returns:
        Synthesized Template model.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    state = PipelineState(work_dir)

    if not resume:
        state.init(source)
    elif not state.completed_stages:
        state.init(source)

    logger.info("Pipeline started: source=%s, resume=%s", source, resume)

    # ── Stage 1: Ingest ──
    if state.should_skip("ingest") and resume:
        logger.info("Skipping ingest (already completed)")
        ingest_data = state.get_stage_data("ingest")
        from reels.models.metadata import IngestResult
        ingest_result = IngestResult(**ingest_data)
    else:
        state.mark_started("ingest")
        ingest_result = ingest_video(source, work_dir, config)
        state.mark_complete("ingest", ingest_result.model_dump())

    # ── Stage 2: Segmentation ──
    if state.should_skip("segmentation") and resume:
        logger.info("Skipping segmentation (already completed)")
        seg_data = state.get_stage_data("segmentation")
        from reels.models.shot import SegmentationResult
        seg_result = SegmentationResult(**seg_data)
    else:
        state.mark_started("segmentation")
        seg_result = segment_video(
            video_path=Path(ingest_result.normalized_video),
            metadata=ingest_result.metadata,
            work_dir=work_dir,
            config=config,
        )
        state.mark_complete("segmentation", seg_result.model_dump())

    # ── Stage 3: Analysis ──
    if state.should_skip("analysis") and resume:
        logger.info("Skipping analysis (already completed)")
        analysis_results = state.get_stage_data("analysis")
    else:
        state.mark_started("analysis")
        ctx = AnalysisContext(
            video_path=Path(ingest_result.normalized_video),
            audio_path=Path(ingest_result.audio_path) if ingest_result.audio_path else None,
            work_dir=work_dir,
            metadata=ingest_result.metadata,
            config=config,
        )

        runner = AnalysisRunner()
        runner.register(PlaceAnalyzer(config))
        runner.register(CameraAnalyzer(config))
        runner.register(SubtitleAnalyzer(config))
        runner.register(SpeechAnalyzer(config))
        runner.register(RhythmAnalyzer(config))

        analysis_results = runner.run_all(seg_result.shots, ctx)

        # Serialize for state persistence
        serialized = {}
        for name, results_list in analysis_results.items():
            serialized[name] = [r.model_dump() for r in results_list]
        state.mark_complete("analysis", serialized)

    # ── Stage 4: Synthesis ──
    if state.should_skip("synthesis") and resume:
        logger.info("Skipping synthesis (already completed)")
        synthesis_data = state.get_stage_data("synthesis")
        output_path = Path(synthesis_data["output_path"])
        template = Template.model_validate_json(output_path.read_text())
    else:
        state.mark_started("synthesis")
        template = synthesize_template(
            shots=seg_result.shots,
            metadata=ingest_result.metadata,
            analysis_results=analysis_results,
            config=config,
            source_url=source if source.startswith("http") else None,
        )

        # Save template JSON to work_dir
        output_path = work_dir / f"{template.template_id}.json"
        output_path.write_text(template.model_dump_json(indent=2))
        logger.info("Template saved: %s", output_path)

        state.mark_complete("synthesis", {"template_id": template.template_id, "output_path": str(output_path)})

    # Archive template + keyframes to persistent storage
    archive_base = config.get("storage", {}).get("templates_dir", "./data/templates")
    archiver = TemplateArchiver(Path(archive_base))
    archive_path = archiver.archive(
        template, work_dir,
        source_video=source if not source.startswith("http") else None,
    )
    logger.info("Template archived: %s", archive_path)

    # Optionally save to DB
    if save_to_db:
        db_path = config.get("db", {}).get("path", "./data/templates.db")
        repo = TemplateRepository(db_path)
        repo.save(template, archive_path=str(archive_path))
        repo.close()
        logger.info("Template saved to DB: %s", template.template_id)

    logger.info("Pipeline complete: %s (%d shots, %.1fs)", template.template_id, template.shot_count, template.total_duration_sec)
    return template
