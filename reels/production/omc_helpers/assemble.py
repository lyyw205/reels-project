"""Storyboard assembly: combine shot_plan + copies into a storyboard.

Extracted from ``CreativeTeamAgent._assemble_storyboard`` and
``CreativeTeamAgent._post_validate_copies`` for reuse by OMC team agents.

CLI usage::

    python -m reels.production.omc_helpers.assemble \\
        --shot-plan shot_plan.json \\
        --copies copies.json \\
        --features features.json \\
        --context context.json \\
        --output-dir /path/to/workdir \\
        [--project-id shorts_abc12345]

Outputs:
    <output_dir>/storyboard.json — assembled Storyboard
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def post_validate_copies(
    copies: list["ShotCopy"],
    shot_plan: "ShotPlan",
    features: list["VerifiedFeature"],
) -> list["ShotCopy"]:
    """Hard validation: check and sanitize factual claims via CopyWriter rules.

    Modifies copies in-place and returns them.
    """
    from reels.production.copy_writer import CopyWriter
    from reels.production.models import CaptionLine, ClaimLevel

    copy_writer = CopyWriter()
    feature_map = {f.tag_en: f for f in features}

    for i, (copy, shot) in enumerate(zip(copies, shot_plan.shots)):
        feat = feature_map.get(shot.feature_tag) if shot.feature_tag else None
        if not feat or feat.claim_level == ClaimLevel.CONFIRMED:
            continue

        claim_level = feat.claim_level

        # Validate hook_line
        if copy.hook_line:
            violations = copy_writer.check_factual_claims(copy.hook_line, claim_level)
            if violations:
                copy.hook_line = copy_writer.sanitize_factual(
                    copy.hook_line, claim_level
                )

        # Validate caption lines
        for j, caption in enumerate(copy.caption_lines):
            violations = copy_writer.check_factual_claims(caption.text, claim_level)
            if violations:
                copy.caption_lines[j] = CaptionLine(
                    text=copy_writer.sanitize_factual(caption.text, claim_level),
                    start_sec=caption.start_sec,
                    end_sec=caption.end_sec,
                    position=caption.position,
                )

    return copies


def assemble_storyboard(
    project_id: str,
    input_data: "AccommodationInput",
    features: list["VerifiedFeature"],
    shot_plan: "ShotPlan",
    copies: list["ShotCopy"],
    match_result: Any = None,
) -> "Storyboard":
    """Assemble PlannedShots + ShotCopies into a Storyboard.

    Args:
        project_id: Unique project identifier.
        input_data: Accommodation input with images and metadata.
        features: Verified features for claim level resolution.
        shot_plan: Shot plan from the director agent.
        copies: Per-shot copy from the writer agent.
        match_result: Optional MatchResult from template matching.

    Returns:
        Assembled Storyboard with all shots populated.
    """
    from reels.production.models import Storyboard, StoryboardShot

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


def main() -> None:
    """CLI entry point: assemble storyboard from JSON files."""
    parser = argparse.ArgumentParser(
        description="Assemble storyboard from shot_plan + copies + features"
    )
    parser.add_argument("--shot-plan", required=True, help="shot_plan.json path")
    parser.add_argument("--copies", required=True, help="copies.json path")
    parser.add_argument("--features", required=True, help="features.json path")
    parser.add_argument("--context", required=True, help="context.json path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--project-id",
        default=None,
        help="Project ID (auto-generated if not specified)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from reels.production.creative_team.models import ShotPlan
    from reels.production.models import AccommodationInput, ShotCopy, VerifiedFeature

    shot_plan = ShotPlan.model_validate_json(Path(args.shot_plan).read_text())
    copies_data = json.loads(Path(args.copies).read_text())
    copies = [ShotCopy.model_validate(c) for c in copies_data]
    features_data = json.loads(Path(args.features).read_text())
    features = [VerifiedFeature.model_validate(f) for f in features_data]
    input_data = AccommodationInput.model_validate_json(Path(args.context).read_text())

    project_id = args.project_id or f"shorts_{uuid.uuid4().hex[:8]}"

    # Post-validate copies (factual claims)
    copies = post_validate_copies(copies, shot_plan, features)

    # Assemble storyboard
    storyboard = assemble_storyboard(
        project_id=project_id,
        input_data=input_data,
        features=features,
        shot_plan=shot_plan,
        copies=copies,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    storyboard_path = output_dir / "storyboard.json"
    storyboard_path.write_text(storyboard.model_dump_json(indent=2))

    print(f"storyboard.json: {storyboard_path}")
    print(f"Shots: {len(storyboard.shots)}")
    print(f"Duration: {storyboard.total_duration_sec:.1f}s")


if __name__ == "__main__":
    main()
