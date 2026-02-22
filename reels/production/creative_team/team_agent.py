"""CreativeTeamAgent — orchestrator for the multi-agent shorts production pipeline."""

from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from reels.production.claim_gate import ClaimGate
from reels.production.copy_writer import CopyWriter
from reels.production.creative_team.llm import CreativeLLM
from reels.production.creative_team.models import QAIssue, QAReport
from reels.production.creative_team.planner import CreativePlanner
from reels.production.creative_team.director import ProducerDirector
from reels.production.creative_team.writer import CreativeWriter
from reels.production.creative_team.reviewer import QAReviewer
from reels.production.feature_extractor import FeatureExtractor
from reels.production.models import (
    AccommodationInput,
    CaptionLine,
    ClaimLevel,
    ProductionResult,
    ShotCopy,
    Storyboard,
    StoryboardShot,
    VerifiedFeature,
)
from reels.production.web_verifier import WebVerifier

logger = logging.getLogger(__name__)


class CreativeTeamAgent:
    """Multi-agent orchestrator for creative shorts production.

    Wraps the existing pipeline (FeatureExtractor, ClaimGate, WebVerifier)
    with a creative team layer (Planner -> Director -> Writer -> Reviewer).

    Usage:
        agent = CreativeTeamAgent(config=config)
        result = await agent.produce(input_data, output_dir)
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        repo: Any = None,
        archiver: Any = None,
    ) -> None:
        cfg = config or {}
        team_cfg = cfg.get("production", {}).get("creative_team", {})
        self._max_revisions = team_cfg.get("max_revisions", 2)

        # Creative team agents (share one CreativeLLM instance)
        self._llm = CreativeLLM(cfg)
        self._planner = CreativePlanner(self._llm, cfg)
        self._director = ProducerDirector(self._llm, cfg)
        self._writer = CreativeWriter(self._llm, cfg)
        self._reviewer = QAReviewer(self._llm, cfg)

        # Existing pipeline modules (reused as-is)
        self.feature_extractor = FeatureExtractor(cfg)
        self.claim_gate = ClaimGate(cfg)
        self.copy_writer = CopyWriter(cfg)  # for check_factual_claims()
        self.web_verifier = WebVerifier(cfg)

        # Optional modules
        self.template_matcher = None
        if repo is not None:
            from reels.production.template_matcher import TemplateMatcher

            self.template_matcher = TemplateMatcher(repo, archiver)

        self.render_spec_gen = None
        try:
            from reels.production.render_spec import RenderSpecGenerator

            self.render_spec_gen = RenderSpecGenerator(cfg)
        except ImportError:
            pass

    async def produce(
        self,
        input_data: AccommodationInput,
        output_dir: Path | None = None,
    ) -> ProductionResult:
        """Run the full creative team production pipeline.

        Phase 0: Feature extraction + claim evaluation (existing modules)
        Phase 1: Creative planning (Planner agent)
        Phase 2: Shot direction + copy writing (Director -> Writer, sequential)
        Phase 3: QA review with feedback loop (Reviewer, max 2 revisions)
        Phase 4: Render spec generation (existing module)
        """
        project_id = f"shorts_{uuid.uuid4().hex[:8]}"

        try:
            # ── Phase 0: Automated pre-processing (existing pipeline) ──
            logger.info("[Phase 0] Feature extraction started")
            features = await self.feature_extractor.extract(
                input_data.images, input_data
            )
            if not features:
                return ProductionResult(
                    project_id=project_id,
                    status="failed",
                    errors=["No features extracted from images"],
                )

            verified = self.claim_gate.evaluate(features)

            if input_data.name and self.web_verifier.enabled:
                verified = await self.web_verifier.verify(verified, input_data.name)
                verified = self.claim_gate.re_evaluate(verified)

            # Template matching (optional)
            match_result = None
            template = None
            if self.template_matcher:
                match_result = self.template_matcher.find_best(verified, input_data)
                if match_result and hasattr(self.template_matcher, "archiver") and self.template_matcher.archiver:
                    template = self.template_matcher.archiver.get_template(
                        match_result.template_id
                    )

            logger.info(
                "[Phase 0] Done: %d features, %d verified",
                len(features),
                len(verified),
            )

            # ── Phase 1: Creative planning ──
            logger.info("[Phase 1] Creative planning started")
            brief = await self._planner.plan(verified, input_data, match_result)
            logger.info(
                "[Phase 1] Brief: structure=%s, heroes=%s",
                brief.narrative_structure,
                brief.hero_features,
            )

            # ── Phase 2: Direction + Copy (sequential) ──
            logger.info("[Phase 2] Direction started")

            # Step 2a: PD creates shot plan
            shot_plan = await self._director.direct(
                brief, verified, list(input_data.images), template
            )
            logger.info(
                "[Phase 2a] ShotPlan: %d shots, %.1fs",
                len(shot_plan.shots),
                shot_plan.total_duration_sec,
            )

            # Step 2b: Writer creates per-shot copy
            copies = await self._writer.write(brief, shot_plan, verified, input_data)
            logger.info("[Phase 2b] Copies: %d items", len(copies))

            # Step 2c: Post-validation (hard guarantee via CopyWriter)
            copies = self._post_validate_copies(copies, shot_plan, verified)

            # Step 2d: Assemble storyboard
            storyboard = self._assemble_storyboard(
                project_id, input_data, verified, shot_plan, copies, match_result
            )
            logger.info("[Phase 2d] Storyboard assembled: %d shots", len(storyboard.shots))

            # ── Phase 3: QA review with feedback loop ──
            logger.info("[Phase 3] QA review started")
            for revision_round in range(self._max_revisions + 1):
                qa_report = await self._reviewer.review(
                    storyboard, verified, revision_round
                )
                logger.info(
                    "[Phase 3] Round %d: verdict=%s, issues=%d",
                    revision_round,
                    qa_report.verdict,
                    len(qa_report.issues),
                )

                if qa_report.verdict == "PASS":
                    break

                if revision_round >= self._max_revisions:
                    logger.warning(
                        "[Phase 3] Max revisions (%d) reached, forcing PASS",
                        self._max_revisions,
                    )
                    break

                # Targeted revision based on responsible_agent
                shot_plan, copies = await self._apply_revisions(
                    qa_report, brief, shot_plan, copies, verified, input_data
                )

                # Re-validate and re-assemble
                copies = self._post_validate_copies(copies, shot_plan, verified)
                storyboard = self._assemble_storyboard(
                    project_id, input_data, verified, shot_plan, copies, match_result
                )

            # ── Phase 4: Render spec generation ──
            render_spec = None
            if self.render_spec_gen:
                render_spec = self.render_spec_gen.generate(storyboard)

            # Save outputs
            if output_dir:
                try:
                    self._save_outputs(output_dir, storyboard, render_spec, verified)
                except Exception as save_err:
                    logger.warning("Failed to save outputs: %s", save_err)

            return ProductionResult(
                project_id=project_id,
                status="complete",
                storyboard=storyboard,
                render_spec=render_spec,
                features=verified,
                output_dir=str(output_dir) if output_dir else "",
            )

        except Exception as e:
            logger.error("Creative team production failed: %s", e)
            return ProductionResult(
                project_id=project_id,
                status="failed",
                errors=[str(e)],
            )

    def _post_validate_copies(
        self,
        copies: list[ShotCopy],
        shot_plan: "ShotPlan",
        features: list[VerifiedFeature],
    ) -> list[ShotCopy]:
        """Hard validation: check factual claims via CopyWriter rules."""
        feature_map = {f.tag_en: f for f in features}

        for i, (copy, shot) in enumerate(zip(copies, shot_plan.shots)):
            feat = feature_map.get(shot.feature_tag) if shot.feature_tag else None
            if not feat or feat.claim_level == ClaimLevel.CONFIRMED:
                continue

            claim_level = feat.claim_level

            # Validate hook_line
            if copy.hook_line:
                violations = self.copy_writer.check_factual_claims(
                    copy.hook_line, claim_level
                )
                if violations:
                    copy.hook_line = self.copy_writer.sanitize_factual(
                        copy.hook_line, claim_level
                    )

            # Validate caption lines
            for j, caption in enumerate(copy.caption_lines):
                violations = self.copy_writer.check_factual_claims(
                    caption.text, claim_level
                )
                if violations:
                    copy.caption_lines[j] = CaptionLine(
                        text=self.copy_writer.sanitize_factual(
                            caption.text, claim_level
                        ),
                        start_sec=caption.start_sec,
                        end_sec=caption.end_sec,
                        position=caption.position,
                    )

        return copies

    def _assemble_storyboard(
        self,
        project_id: str,
        input_data: AccommodationInput,
        features: list[VerifiedFeature],
        shot_plan: "ShotPlan",
        copies: list[ShotCopy],
        match_result: Any = None,
    ) -> Storyboard:
        """Assemble PlannedShots + ShotCopies into a Storyboard."""
        feature_map = {f.tag_en: f for f in features}
        images = list(input_data.images)
        shots: list[StoryboardShot] = []
        current_sec = 0.0

        for planned, copy in zip(shot_plan.shots, copies):
            # Resolve asset path
            img_idx = min(planned.image_index, len(images) - 1) if images else 0
            asset_path = str(images[img_idx]) if images else ""

            # Resolve claim level
            feat = feature_map.get(planned.feature_tag) if planned.feature_tag else None
            claim_level = feat.claim_level if feat else None

            sb_shot = planned.to_storyboard_shot(
                copy=copy,
                asset_path=asset_path,
                claim_level=claim_level,
                start_sec=current_sec,
                end_sec=current_sec + planned.duration_sec,
            )
            shots.append(sb_shot)
            current_sec += planned.duration_sec

        return Storyboard(
            project_id=project_id,
            accommodation_name=input_data.name,
            target_audience=input_data.target_audience,
            features=features,
            template_ref=match_result.template_id if match_result else None,
            total_duration_sec=current_sec,
            shots=shots,
        )

    async def _apply_revisions(
        self,
        qa_report: QAReport,
        brief: "CreativeBrief",
        shot_plan: "ShotPlan",
        copies: list[ShotCopy],
        features: list[VerifiedFeature],
        input_data: AccommodationInput,
    ) -> tuple["ShotPlan", list[ShotCopy]]:
        """Route QA issues to responsible agents and apply targeted revisions."""
        # Group issues by responsible agent
        by_agent: dict[str, list[QAIssue]] = defaultdict(list)
        for issue in qa_report.issues:
            by_agent[issue.responsible_agent].append(issue)

        # Cascade: planner → director → writer
        if "planner" in by_agent:
            logger.info("Revising: planner (%d issues)", len(by_agent["planner"]))
            brief = await self._planner.revise(brief, by_agent["planner"])
            # Planner change cascades: re-run director and writer
            shot_plan = await self._director.direct(
                brief, features, list(input_data.images), None
            )
            copies = await self._writer.write(brief, shot_plan, features, input_data)

        elif "director" in by_agent:
            logger.info("Revising: director (%d issues)", len(by_agent["director"]))
            shot_plan = await self._director.revise(shot_plan, by_agent["director"])
            # Director change: re-run writer too (shot structure changed)
            copies = await self._writer.write(brief, shot_plan, features, input_data)

        if "writer" in by_agent and "planner" not in by_agent and "director" not in by_agent:
            logger.info("Revising: writer (%d issues)", len(by_agent["writer"]))
            copies = await self._writer.revise(
                copies, shot_plan, by_agent["writer"]
            )

        return shot_plan, copies

    def _save_outputs(
        self,
        output_dir: Path,
        storyboard: Storyboard,
        render_spec: Any,
        features: list[VerifiedFeature],
    ) -> None:
        """Save all output files to directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "features.json").write_text(
            json.dumps(
                [f.model_dump() for f in features], ensure_ascii=False, indent=2
            )
        )

        (output_dir / "storyboard.json").write_text(
            storyboard.model_dump_json(indent=2)
        )

        if render_spec:
            (output_dir / "render_spec.json").write_text(
                render_spec.model_dump_json(indent=2)
            )
            if render_spec.captions_srt:
                (output_dir / "captions.srt").write_text(render_spec.captions_srt)
            if render_spec.vo_script:
                (output_dir / "vo.txt").write_text(render_spec.vo_script)
