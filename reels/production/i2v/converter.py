"""Image-to-Video converter using Replicate API (Wan2.1).

Requires REPLICATE_API_TOKEN environment variable.

Usage::

    from reels.production.i2v.converter import I2VConverter

    converter = I2VConverter()
    video_path = await converter.convert(
        image_path="path/to/image.jpg",
        prompt="water flowing from wooden bucket",
        output_path="path/to/output.mp4",
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default I2V model on Replicate (minimax/video-01 is in the free tier)
DEFAULT_MODEL = "minimax/video-01"


class I2VConverter:
    """Convert static images to short video clips via Replicate API."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("production", {}).get("i2v", {})
        self.model: str = cfg.get("model", DEFAULT_MODEL)
        self.default_num_frames: int = cfg.get("num_frames", 81)  # ~5s at 16fps
        self.default_guidance_scale: float = cfg.get("guidance_scale", 5.0)
        self._token = os.environ.get("REPLICATE_API_TOKEN", "")

    @property
    def available(self) -> bool:
        """Check if Replicate API token is configured."""
        return bool(self._token)

    async def convert(
        self,
        image_path: str | Path,
        prompt: str,
        output_path: str | Path,
        duration_sec: float = 3.0,
    ) -> Path:
        """Convert a single image to a video clip.

        Args:
            image_path: Path to source image.
            prompt: Motion description prompt.
            output_path: Where to save the generated video.
            duration_sec: Target duration (approximate, model-dependent).

        Returns:
            Path to the generated video file.

        Raises:
            RuntimeError: If API call fails or token is missing.
        """
        if not self.available:
            raise RuntimeError(
                "REPLICATE_API_TOKEN not set. "
                "Get one at https://replicate.com/account/api-tokens"
            )

        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "I2V converting: %s -> %s (prompt: %s)",
            image_path.name,
            output_path.name,
            prompt[:60],
        )

        try:
            import replicate  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "replicate package not installed. Run: pip install replicate"
            )

        # Calculate frames based on duration (Wan2.1 generates at ~16fps)
        num_frames = max(33, min(int(duration_sec * 16), 81))

        # Run in thread pool to avoid blocking async loop
        output_url = await asyncio.to_thread(
            self._run_replicate,
            image_path,
            prompt,
            num_frames,
        )

        # Download result
        await asyncio.to_thread(self._download, output_url, output_path)
        logger.info("I2V complete: %s (%.1f MB)", output_path.name, output_path.stat().st_size / 1e6)

        return output_path

    def _run_replicate(
        self,
        image_path: Path,
        prompt: str,
        num_frames: int,
        max_retries: int = 3,
    ) -> str:
        """Synchronous Replicate API call with retry on rate limit."""
        import replicate

        for attempt in range(max_retries):
            try:
                with open(image_path, "rb") as f:
                    input_params: dict[str, Any] = {
                        "prompt": prompt,
                        "prompt_optimizer": True,
                    }
                    # minimax/video-01 uses first_frame_image; Wan models use image
                    if "minimax" in self.model:
                        input_params["first_frame_image"] = f
                    else:
                        input_params["image"] = f
                        input_params["max_area"] = "832x480"
                        input_params["num_frames"] = num_frames
                        input_params["guidance_scale"] = self.default_guidance_scale
                        input_params["num_inference_steps"] = 30

                    output = replicate.run(
                        self.model,
                        input=input_params,
                    )

                # Output can be a FileOutput, URL string, or iterator
                if hasattr(output, "url"):
                    return str(output.url)
                if isinstance(output, str):
                    return output
                # Iterator case — take first item
                if hasattr(output, "__iter__"):
                    for item in output:
                        if hasattr(item, "url"):
                            return str(item.url)
                        return str(item)

                raise RuntimeError(f"Unexpected Replicate output type: {type(output)}")

            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate" in err_str.lower()
                if is_rate_limit and attempt < max_retries - 1:
                    wait = 30 * (2 ** attempt)  # 30s, 60s, 120s
                    logger.warning(
                        "Rate limited (attempt %d/%d), waiting %ds...",
                        attempt + 1, max_retries, wait,
                    )
                    time.sleep(wait)
                    continue
                raise

        raise RuntimeError(f"_run_replicate: exhausted {max_retries} retries without returning")

    def _download(self, url: str, output_path: Path) -> None:
        """Download video from URL."""
        urllib.request.urlretrieve(url, str(output_path))


async def convert_batch(
    converter: I2VConverter,
    candidates: list[dict],
    output_dir: Path,
    base_dir: Path | None = None,
    max_concurrent: int = 1,
    delay_between: float = 10.0,
) -> dict[int, Path]:
    """Convert multiple images to videos sequentially to avoid rate limits.

    Args:
        converter: I2VConverter instance.
        candidates: List of dicts with shot_id, image_path, prompt, duration_sec.
        output_dir: Directory for output videos.
        base_dir: Base directory to resolve relative image paths.
        max_concurrent: Max parallel API calls (default 1 for free tier).
        delay_between: Seconds to wait between conversions.

    Returns:
        Dict mapping shot_id to generated video path.
    """
    results: dict[int, Path] = {}

    for i, candidate in enumerate(candidates):
        shot_id = candidate["shot_id"]
        image_path = Path(candidate["image_path"])
        if base_dir and not image_path.is_absolute():
            image_path = base_dir / image_path

        output_path = output_dir / f"shot_{shot_id}_i2v.mp4"

        try:
            path = await converter.convert(
                image_path=image_path,
                prompt=candidate["prompt"],
                output_path=output_path,
                duration_sec=candidate.get("duration_sec", 3.0),
            )
            results[shot_id] = path
        except Exception as e:
            logger.error("I2V failed for shot %d: %s", shot_id, e)

        # Wait between requests to avoid rate limiting
        if i < len(candidates) - 1:
            logger.info("Waiting %.0fs before next conversion...", delay_between)
            await asyncio.sleep(delay_between)

    return results
