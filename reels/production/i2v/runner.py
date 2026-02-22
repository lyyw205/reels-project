"""I2V pipeline runner: select shots, convert images to video, update storyboard.

CLI usage::

    python -m reels.production.i2v.runner \\
        --storyboard output/okinawa/storyboard.json \\
        --output-dir output/okinawa \\
        [--max-shots 4] [--base-dir .]

Outputs:
    <output_dir>/shot_<N>_i2v.mp4   — generated video clips
    <output_dir>/storyboard_i2v.json — updated storyboard with video asset paths
    <output_dir>/i2v_manifest.json   — conversion manifest
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


async def run_i2v_pipeline(
    storyboard_path: Path,
    output_dir: Path,
    base_dir: Path | None = None,
    max_shots: int = 4,
    config: dict[str, Any] | None = None,
) -> dict:
    """Run the full I2V pipeline.

    1. Load storyboard
    2. Select shots for I2V conversion
    3. Convert images to video via Replicate
    4. Update storyboard with video asset paths
    5. Save updated storyboard + manifest

    Returns:
        Dict with conversion results summary.
    """
    from reels.production.i2v.converter import I2VConverter, convert_batch
    from reels.production.i2v.motion_selector import select_shots_for_i2v

    # 1. Load storyboard
    storyboard_data = json.loads(storyboard_path.read_text())

    # 2. Select shots
    candidates = select_shots_for_i2v(storyboard_data, max_shots=max_shots)

    if not candidates:
        logger.info("No shots selected for I2V conversion")
        return {"converted": 0, "shots": []}

    logger.info(
        "Selected %d shots for I2V: %s",
        len(candidates),
        [c.shot_id for c in candidates],
    )

    # 3. Convert
    converter = I2VConverter(config)
    if not converter.available:
        logger.error(
            "REPLICATE_API_TOKEN not set. "
            "Set it with: export REPLICATE_API_TOKEN=r8_..."
        )
        return {"converted": 0, "error": "REPLICATE_API_TOKEN not set"}

    candidate_dicts = [
        {
            "shot_id": c.shot_id,
            "image_path": c.image_path,
            "prompt": c.prompt,
            "duration_sec": c.duration_sec,
        }
        for c in candidates
    ]

    video_dir = output_dir / "i2v_clips"
    results = await convert_batch(
        converter,
        candidate_dicts,
        video_dir,
        base_dir=base_dir,
        max_concurrent=2,
    )

    # 4. Update storyboard
    updated_storyboard = _update_storyboard(storyboard_data, results, video_dir)

    # 5. Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    updated_path = output_dir / "storyboard_i2v.json"
    updated_path.write_text(
        json.dumps(updated_storyboard, ensure_ascii=False, indent=2)
    )

    manifest = {
        "converted": len(results),
        "shots": [
            {
                "shot_id": sid,
                "video_path": str(vpath),
                "original_image": next(
                    (c.image_path for c in candidates if c.shot_id == sid), ""
                ),
                "prompt": next(
                    (c.prompt for c in candidates if c.shot_id == sid), ""
                ),
            }
            for sid, vpath in results.items()
        ],
        "unchanged_shots": [
            c.shot_id
            for c in candidates
            if c.shot_id not in results
        ],
    }

    manifest_path = output_dir / "i2v_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )

    logger.info(
        "I2V pipeline complete: %d/%d shots converted",
        len(results),
        len(candidates),
    )

    return manifest


def _update_storyboard(
    storyboard_data: dict,
    video_map: dict[int, Path],
    video_dir: Path,
) -> dict:
    """Update storyboard shots with video asset paths."""
    updated = json.loads(json.dumps(storyboard_data))  # deep copy

    for shot in updated.get("shots", []):
        shot_id = shot.get("shot_id")
        if shot_id in video_map:
            shot["asset_type"] = "video"
            shot["asset_path"] = str(video_map[shot_id])
            shot["original_image"] = shot.get("asset_path", "")

    return updated


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert selected storyboard shots from images to video"
    )
    parser.add_argument(
        "--storyboard", required=True, help="storyboard.json path"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory"
    )
    parser.add_argument(
        "--base-dir", default=".", help="Base directory for resolving image paths"
    )
    parser.add_argument(
        "--max-shots", type=int, default=4, help="Max shots to convert"
    )
    parser.add_argument(
        "--config", default=None, help="Config YAML path"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    config: dict[str, Any] = {}
    if args.config:
        from reels.config import get_config
        config = get_config(Path(args.config))

    result = asyncio.run(
        run_i2v_pipeline(
            storyboard_path=Path(args.storyboard),
            output_dir=Path(args.output_dir),
            base_dir=Path(args.base_dir),
            max_shots=args.max_shots,
            config=config,
        )
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
