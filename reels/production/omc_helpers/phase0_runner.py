"""Phase 0 runner: feature extraction, claim gating, and web verification.

Runs the automated pre-processing phase of the production pipeline and saves
results as JSON files for OMC team agents to consume.

CLI usage::

    python -m reels.production.omc_helpers.phase0_runner \\
        --images img1.jpg img2.jpg \\
        --output-dir /path/to/workdir \\
        [--name "숙소명"] [--region "지역"] [--target couple] [--no-web]

Outputs:
    <output_dir>/features.json   — list of VerifiedFeature dicts
    <output_dir>/context.json    — AccommodationInput dict
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _build_input(
    images: list[str],
    name: str | None = None,
    region: str | None = None,
    target: str = "couple",
) -> "AccommodationInput":
    from reels.production.models import AccommodationInput, TargetAudience

    return AccommodationInput(
        name=name,
        region=region,
        target_audience=TargetAudience(target),
        images=[Path(img) for img in images],
    )


async def run_phase0(
    input_data: "AccommodationInput",
    config: dict[str, Any] | None = None,
    no_web: bool = False,
) -> tuple[list["VerifiedFeature"], "AccommodationInput"]:
    """Execute Phase 0: feature extraction + claim gate + optional web verification.

    Args:
        input_data: Accommodation input with images and metadata.
        config: Pipeline configuration dict.
        no_web: If True, skip web verification.

    Returns:
        Tuple of (verified_features, input_data).

    Raises:
        RuntimeError: If no features are extracted.
    """
    from reels.production.claim_gate import ClaimGate
    from reels.production.feature_extractor import FeatureExtractor
    from reels.production.web_verifier import WebVerifier

    cfg = config or {}

    if no_web:
        cfg.setdefault("production", {}).setdefault("web_verification", {})["enabled"] = False

    extractor = FeatureExtractor(cfg)
    claim_gate = ClaimGate(cfg)
    web_verifier = WebVerifier(cfg)

    logger.info("Phase 0: extracting features from %d images", len(input_data.images))
    features = await extractor.extract(input_data.images, input_data)

    if not features:
        raise RuntimeError("No features extracted from images")

    verified = claim_gate.evaluate(features)
    logger.info("Phase 0: %d features, %d verified", len(features), len(verified))

    if input_data.name and web_verifier.enabled:
        verified = await web_verifier.verify(verified, input_data.name)
        verified = claim_gate.re_evaluate(verified)
        logger.info("Phase 0: web verification complete")

    return verified, input_data


def save_phase0_outputs(
    output_dir: Path,
    features: list["VerifiedFeature"],
    input_data: "AccommodationInput",
) -> tuple[Path, Path]:
    """Save Phase 0 results to JSON files.

    Returns:
        Tuple of (features_path, context_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = output_dir / "features.json"
    features_path.write_text(
        json.dumps(
            [f.model_dump(mode="json") for f in features],
            ensure_ascii=False,
            indent=2,
        )
    )

    context_path = output_dir / "context.json"
    context_path.write_text(
        input_data.model_dump_json(indent=2)
    )

    return features_path, context_path


def main() -> None:
    """CLI entry point for Phase 0 runner."""
    parser = argparse.ArgumentParser(
        description="Run Phase 0: feature extraction + claim gating"
    )
    parser.add_argument(
        "--images", nargs="+", required=True, help="Image file paths"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for JSON files"
    )
    parser.add_argument("--name", default=None, help="Accommodation name")
    parser.add_argument("--region", default=None, help="Region")
    parser.add_argument(
        "--target",
        choices=["couple", "family", "solo", "friends"],
        default="couple",
        help="Target audience",
    )
    parser.add_argument(
        "--no-web", action="store_true", help="Skip web verification"
    )
    parser.add_argument(
        "--config", default=None, help="Config YAML path"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config: dict[str, Any] = {}
    if args.config:
        from reels.config import get_config
        config = get_config(Path(args.config))

    input_data = _build_input(
        images=args.images,
        name=args.name,
        region=args.region,
        target=args.target,
    )

    output_dir = Path(args.output_dir)

    try:
        verified, input_data = asyncio.run(
            run_phase0(input_data, config, no_web=args.no_web)
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    features_path, context_path = save_phase0_outputs(output_dir, verified, input_data)
    print(f"features.json: {features_path}")
    print(f"context.json: {context_path}")
    print(f"Total features: {len(verified)}")


if __name__ == "__main__":
    main()
