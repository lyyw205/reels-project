"""Select which shots benefit from I2V conversion and generate motion prompts.

Analyzes feature tags, shot roles, and image content hints to decide which
shots should be converted from static images to short video clips.
"""

from __future__ import annotations

from dataclasses import dataclass

# Feature tags that naturally benefit from motion
_MOTION_FEATURES: dict[str, str] = {
    "japanese_round_bath": "gentle water surface ripples in a round stone bath, soft steam rising, warm ambient lighting",
    "rain_shower": "water streaming from rain shower head, droplets splashing, steam rising",
    "aroma_candle": "candle flame gently flickering, warm glow casting soft shadows",
    "paper_lantern": "paper lantern gently swaying, soft light casting moving shadows on walls",
    "outdoor_bath": "steam rising from outdoor bath, gentle water surface movement, natural breeze",
    "pool": "gentle water ripples in pool, light reflections dancing on surface",
    "fireplace": "warm fire crackling, flames dancing, embers floating upward",
    "ocean_view": "ocean waves gently rolling, sunlight sparkling on water surface",
    "mountain_view": "clouds slowly drifting past mountains, gentle breeze through trees",
    "garden": "leaves gently swaying in breeze, flowers slightly moving",
}

# Shot roles that benefit more from motion
_MOTION_ROLES = {"hook", "feature"}


@dataclass
class I2VCandidate:
    """A shot selected for I2V conversion."""

    shot_id: int
    image_path: str
    prompt: str
    duration_sec: float
    priority: int  # lower = higher priority


def select_shots_for_i2v(
    storyboard_data: dict,
    max_shots: int = 4,
) -> list[I2VCandidate]:
    """Analyze storyboard and select shots that benefit most from I2V.

    Selection criteria (in priority order):
    1. Feature tag matches known motion features (water, fire, steam)
    2. Shot role is hook or feature (higher visual impact)
    3. Longer shots benefit more (2s+ shots)

    Args:
        storyboard_data: Parsed storyboard.json dict.
        max_shots: Maximum number of shots to convert.

    Returns:
        List of I2VCandidate sorted by priority.
    """
    candidates: list[I2VCandidate] = []

    for shot in storyboard_data.get("shots", []):
        feature_tag = shot.get("feature_tag", "")
        role = shot.get("role", "")

        # Check if feature tag has a known motion prompt
        if feature_tag not in _MOTION_FEATURES:
            continue

        # Skip CTA shots (usually text-heavy, motion distracts)
        if role == "cta":
            continue

        # Build context-aware prompt
        base_prompt = _MOTION_FEATURES[feature_tag]
        camera = shot.get("camera_suggestion", "static")
        camera_hint = _camera_to_prompt_hint(camera)
        prompt = f"{base_prompt}, {camera_hint}, cinematic lighting, 4K quality"

        # Priority scoring (lower = better)
        priority = 10
        if role == "hook":
            priority -= 3
        if role == "feature":
            priority -= 2
        if shot.get("duration_sec", 0) >= 2.0:
            priority -= 2

        candidates.append(
            I2VCandidate(
                shot_id=shot["shot_id"],
                image_path=shot.get("asset_path", ""),
                prompt=prompt,
                duration_sec=min(shot.get("duration_sec", 2.0), 3.0),
                priority=priority,
            )
        )

    # Sort by priority and limit
    candidates.sort(key=lambda c: c.priority)
    return candidates[:max_shots]


def _camera_to_prompt_hint(camera: str) -> str:
    """Convert camera suggestion to motion prompt hint."""
    hints = {
        "push_in": "slow camera push in",
        "pull_out": "slow camera pull out revealing the scene",
        "pan_right": "slow camera pan to the right",
        "pan_left": "slow camera pan to the left",
        "tilt_up": "slow camera tilt upward",
        "tilt_down": "slow camera tilt downward",
        "gimbal_smooth": "smooth gimbal camera movement",
        "static": "static camera, subtle ambient motion only",
    }
    return hints.get(camera, "static camera")
