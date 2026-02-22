"""Analysis runner: orchestrates all analyzers sequentially with cleanup."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from reels.analysis.base import AnalysisContext
from reels.models.shot import Shot

logger = logging.getLogger(__name__)


class AnalysisRunner:
    """Run registered analyzers sequentially with cleanup between each."""

    def __init__(self) -> None:
        self._analyzers: list[Any] = []

    def register(self, analyzer: Any) -> None:
        """Register an analyzer instance."""
        self._analyzers.append(analyzer)

    def run_all(
        self,
        shots: list[Shot],
        context: AnalysisContext,
    ) -> dict[str, list[BaseModel]]:
        """Run all registered analyzers on all shots.

        Each analyzer runs on all shots, then cleanup() is called to free memory.
        If an analyzer fails, its results are replaced with empty defaults.

        Returns:
            Dict mapping analyzer name to list of results (one per shot).
        """
        results: dict[str, list[BaseModel]] = {}

        for analyzer in self._analyzers:
            name = analyzer.name
            logger.info("Running analyzer: %s (%d shots)", name, len(shots))

            try:
                analyzer_results = analyzer.analyze_batch(shots, context)
                results[name] = analyzer_results
                logger.info("Analyzer %s complete: %d results", name, len(analyzer_results))
            except Exception as e:
                logger.warning("Analyzer %s failed: %s (continuing with empty results)", name, e)
                # Create empty results - use the first result type's default
                results[name] = []
            finally:
                try:
                    analyzer.cleanup()
                except Exception as e:
                    logger.warning("Cleanup failed for %s: %s", name, e)

        return results
