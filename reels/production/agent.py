"""ProductionAgent — main orchestrator for the Shorts Production pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from reels.production.claim_gate import ClaimGate
from reels.production.copy_writer import CopyWriter
from reels.production.feature_extractor import FeatureExtractor
from reels.production.models import (
    AccommodationInput,
    ProductionResult,
)
from reels.production.storyboard_builder import StoryboardBuilder
from reels.production.web_verifier import WebVerifier

logger = logging.getLogger(__name__)


class ProductionAgent:
    """Feature-first Shorts Production Agent - main orchestrator."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        repo=None,
        archiver=None,
    ) -> None:
        cfg = config or {}
        self.feature_extractor = FeatureExtractor(cfg)
        self.claim_gate = ClaimGate(cfg)
        self.copy_writer = CopyWriter(cfg)

        self.template_matcher = None
        if repo is not None:
            from reels.production.template_matcher import TemplateMatcher
            self.template_matcher = TemplateMatcher(repo, archiver)

        self.storyboard_builder = StoryboardBuilder(cfg)
        self.web_verifier = WebVerifier(cfg)

        # RenderSpecGenerator may not be implemented yet — handle ImportError gracefully
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
        """Full production pipeline execution.

        Partial failure allowed -> ProductionResult with status field.
        """
        try:
            # 1. Feature extraction (parallel with semaphore)
            features = await self.feature_extractor.extract(input_data.images, input_data)
            if not features:
                return ProductionResult(
                    project_id=self._gen_id(),
                    status="failed",
                    errors=["No features extracted from images"],
                )

            # 2. Claim Gate (confidence evaluation)
            verified = self.claim_gate.evaluate(features)

            # 3. Web verification (only if name available and enabled)
            if input_data.name and self.web_verifier.enabled:
                verified = await self.web_verifier.verify(verified, input_data.name)
                # Re-evaluate after web verification changes confidence
                verified = self.claim_gate.re_evaluate(verified)

            # 4. Template matching (BEFORE CopyWriter - determines shot_count)
            match_result = None
            template = None
            if self.template_matcher:
                match_result = self.template_matcher.find_best(verified, input_data)

            if match_result:
                shot_count = match_result.shot_count
                # Load actual template for storyboard
                if self.template_matcher and hasattr(self.template_matcher, "archiver") and self.template_matcher.archiver:
                    template = self.template_matcher.archiver.get_template(match_result.template_id)
            else:
                shot_count = 7  # default

            # 5. Copy generation (with confirmed shot_count)
            copies = self.copy_writer.generate(verified, input_data, shot_count)

            # 6. Storyboard assembly
            storyboard = self.storyboard_builder.build(
                verified, copies, template, list(input_data.images), input_data,
            )

            # 7. Render spec generation
            render_spec = None
            if self.render_spec_gen:
                render_spec = self.render_spec_gen.generate(storyboard)

            # 8. Save outputs (isolated — disk failure should not discard results)
            if output_dir:
                try:
                    self._save_outputs(output_dir, storyboard, render_spec, verified)
                except Exception as save_err:
                    logger.warning("Failed to save outputs: %s", save_err)

            return ProductionResult(
                project_id=storyboard.project_id,
                status="complete",
                storyboard=storyboard,
                render_spec=render_spec,
                features=verified,
                output_dir=str(output_dir) if output_dir else "",
            )

        except Exception as e:
            logger.error("Production failed: %s", e)
            return ProductionResult(
                project_id=self._gen_id(),
                status="failed",
                errors=[str(e)],
            )

    def _save_outputs(self, output_dir: Path, storyboard, render_spec, features) -> None:
        """Save all output files to directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # features.json
        (output_dir / "features.json").write_text(
            json.dumps([f.model_dump() for f in features], ensure_ascii=False, indent=2)
        )

        # storyboard.json
        (output_dir / "storyboard.json").write_text(
            storyboard.model_dump_json(indent=2)
        )

        # render_spec.json
        if render_spec:
            (output_dir / "render_spec.json").write_text(
                render_spec.model_dump_json(indent=2)
            )

            # captions.srt
            if render_spec.captions_srt:
                (output_dir / "captions.srt").write_text(render_spec.captions_srt)

            # vo.txt
            if render_spec.vo_script:
                (output_dir / "vo.txt").write_text(render_spec.vo_script)

    @staticmethod
    def _gen_id() -> str:
        import uuid
        return f"shorts_{uuid.uuid4().hex[:8]}"
