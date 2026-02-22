"""Phase 4 runner: storyboard → render spec + SRT captions + VO script.

Wraps ``RenderSpecGenerator`` as a standalone CLI for OMC team agents.

CLI usage::

    python -m reels.production.omc_helpers.render_spec_runner \\
        --storyboard storyboard.json \\
        --output-dir /path/to/workdir \\
        [--config config.yaml]

Outputs:
    <output_dir>/render_spec.json  — Remotion render spec
    <output_dir>/captions.srt      — SRT subtitles (if captions exist)
    <output_dir>/vo.txt            — VO narration script (if present)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reels.production.models import RenderSpec, Storyboard

logger = logging.getLogger(__name__)


def run_render_spec(
    storyboard: "Storyboard",
    config: dict[str, Any] | None = None,
) -> "RenderSpec":
    """Generate a RenderSpec from a Storyboard.

    Args:
        storyboard: Assembled storyboard.
        config: Pipeline configuration dict.

    Returns:
        RenderSpec with asset mappings, SRT captions, VO script, and music spec.
    """
    from reels.production.render_spec import RenderSpecGenerator

    generator = RenderSpecGenerator(config)
    return generator.generate(storyboard)


def save_render_outputs(
    output_dir: Path,
    render_spec: "RenderSpec",
) -> dict[str, Path]:
    """Save render spec outputs to files.

    Returns:
        Dict mapping output type to file path (only includes files that were written).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    spec_path = output_dir / "render_spec.json"
    spec_path.write_text(render_spec.model_dump_json(indent=2))
    written["render_spec"] = spec_path

    if render_spec.captions_srt:
        srt_path = output_dir / "captions.srt"
        srt_path.write_text(render_spec.captions_srt)
        written["captions_srt"] = srt_path

    if render_spec.vo_script:
        vo_path = output_dir / "vo.txt"
        vo_path.write_text(render_spec.vo_script)
        written["vo_script"] = vo_path

    return written


def main() -> None:
    """CLI entry point for Phase 4 render spec generation."""
    parser = argparse.ArgumentParser(
        description="Generate render spec from storyboard"
    )
    parser.add_argument(
        "--storyboard", required=True, help="storyboard.json path"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory"
    )
    parser.add_argument(
        "--config", default=None, help="Config YAML path"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from reels.production.models import Storyboard

    storyboard_path = Path(args.storyboard)
    if not storyboard_path.exists():
        print(f"Error: {storyboard_path} not found", file=sys.stderr)
        sys.exit(1)

    storyboard = Storyboard.model_validate_json(storyboard_path.read_text())

    config: dict[str, Any] = {}
    if args.config:
        from reels.config import get_config
        config = get_config(Path(args.config))

    render_spec = run_render_spec(storyboard, config)
    output_dir = Path(args.output_dir)
    written = save_render_outputs(output_dir, render_spec)

    for kind, path in written.items():
        print(f"{kind}: {path}")
    print(f"Assets: {len(render_spec.assets)} mappings")


if __name__ == "__main__":
    main()
