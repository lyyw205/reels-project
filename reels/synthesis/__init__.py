"""Template synthesis module: assemble analysis results into templates."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from reels.models.metadata import VideoMetadata
from reels.models.shot import Shot
from reels.models.template import Template
from reels.synthesis.assembler import TemplateAssembler
from reels.synthesis.schema_validator import validate_template

logger = logging.getLogger(__name__)


def synthesize_template(
    shots: list[Shot],
    metadata: VideoMetadata,
    analysis_results: dict[str, list[BaseModel]],
    config: dict,
    source_url: str | None = None,
) -> Template:
    """Synthesize a Template from shots and analysis results.

    Args:
        shots: Segmented shots.
        metadata: Video metadata.
        analysis_results: Dict of analyzer name -> list of results per shot.
        config: Pipeline config.
        source_url: Original video URL.

    Returns:
        Validated Template model.
    """
    assembler = TemplateAssembler(config)
    template = assembler.assemble(shots, metadata, analysis_results, source_url)
    validate_template(template)

    logger.info(
        "Synthesized template %s: %d shots, %.1fs",
        template.template_id, template.shot_count, template.total_duration_sec,
    )
    return template
