"""Web verifier — optional fact-checking for accommodation features."""

from __future__ import annotations

import logging
from typing import Any

from reels.production.models import (
    ClaimLevel,
    VerifiedFeature,
    WebEvidence,
)

logger = logging.getLogger(__name__)

_DEFAULT_TRUSTED_DOMAINS = [
    "booking.com", "hotels.com", "naver.com", "kakao.com",
    "trip.com", "agoda.com", "yeogi.com",
]


class WebVerifier:
    """Verify accommodation features via web search (optional)."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("production", {}).get("web_verification", {})
        self.enabled = cfg.get("enabled", True)
        self.max_queries = cfg.get("max_queries", 4)
        self.trusted_domains = cfg.get("trusted_domains", _DEFAULT_TRUSTED_DOMAINS)
        self._client = None  # Lazy httpx client

    async def verify(
        self,
        features: list[VerifiedFeature],
        accommodation_name: str | None = None,
    ) -> list[VerifiedFeature]:
        """Verify features that need web confirmation.

        Only verifies PROBABLE features that are candidates for main usage.
        Skips if no accommodation name or web verification disabled.
        Returns updated features with web evidence.
        """
        if not self.enabled or not accommodation_name:
            logger.info("Web verification skipped (enabled=%s, name=%s)",
                       self.enabled, bool(accommodation_name))
            return features

        result = []
        queries_used = 0

        for i, feature in enumerate(features):
            is_main = i < 2  # Top 2 features are "main"

            if self._should_verify(feature, is_main) and queries_used < self.max_queries:
                try:
                    evidence = await self._search_feature(
                        accommodation_name, feature.tag,
                    )
                    queries_used += 1
                    updated = self._update_with_evidence(feature, evidence)
                    result.append(updated)
                    logger.info(
                        "Verified '%s': %d evidence items found",
                        feature.tag, len(evidence),
                    )
                except Exception as e:
                    logger.warning("Web verification failed for '%s': %s", feature.tag, e)
                    result.append(feature)
            else:
                result.append(feature)

        return result

    def _should_verify(self, feature: VerifiedFeature, is_main: bool) -> bool:
        """Determine if feature needs web verification.

        Rules:
        1. Already CONFIRMED -> skip
        2. SUGGESTIVE -> skip (too uncertain)
        3. PROBABLE + main feature -> verify
        """
        if feature.claim_level == ClaimLevel.CONFIRMED:
            return False
        if feature.claim_level == ClaimLevel.SUGGESTIVE:
            return False
        return is_main  # Only verify PROBABLE main features

    async def _search_feature(
        self, accommodation_name: str, feature_tag: str,
    ) -> list[WebEvidence]:
        """Search web for evidence of a feature at this accommodation."""
        query = f"{accommodation_name} {feature_tag}"

        try:
            client = self._get_client()
            # Use a search API or scrape approach
            # For MVP: simple httpx GET to a search endpoint
            response = await client.get(
                "https://search.naver.com/search.naver",
                params={"query": query},
                timeout=10.0,
            )

            if response.status_code == 200:
                # Parse search results for evidence
                return self._parse_search_results(
                    response.text, accommodation_name, feature_tag,
                )
            return []
        except Exception as e:
            logger.warning("Search failed for '%s': %s", query, e)
            return []

    def _parse_search_results(
        self, html: str, name: str, tag: str,
    ) -> list[WebEvidence]:
        """Extract evidence from search results HTML.

        Looks for the feature tag near the accommodation name in results.
        Returns evidence items with confidence based on match quality.
        """
        evidence = []

        # Simple heuristic: check if both name and tag appear in response
        name_lower = name.lower()
        tag_lower = tag.lower()

        if name_lower in html.lower() and tag_lower in html.lower():
            evidence.append(WebEvidence(
                claim=f"{name} has {tag}",
                url=f"https://search.naver.com/search.naver?query={name}+{tag}",
                snippet=f"Search results mention {tag} at {name}",
                confidence=0.7,
            ))

        return evidence

    def _update_with_evidence(
        self, feature: VerifiedFeature, evidence: list[WebEvidence],
    ) -> VerifiedFeature:
        """Update feature with web evidence."""
        if not evidence:
            return feature

        # Combine existing and new evidence
        all_evidence = list(feature.web_evidence) + evidence

        # Boost confidence based on evidence quality
        avg_evidence_conf = sum(e.confidence for e in evidence) / len(evidence)
        new_confidence = min(1.0, max(feature.confidence, avg_evidence_conf))

        return feature.model_copy(update={
            "confidence": new_confidence,
            "web_evidence": all_evidence,
        })

    def _get_client(self):
        """Lazy-init httpx async client."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(
                    headers={"User-Agent": "ReelsBot/1.0"},
                    follow_redirects=True,
                )
            except ImportError:
                raise RuntimeError("httpx required for web verification")
        return self._client

    async def cleanup(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
