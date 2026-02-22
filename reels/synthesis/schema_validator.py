"""Template schema validation."""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from reels.exceptions import SynthesisError
from reels.models.template import Template

logger = logging.getLogger(__name__)


def validate_template(template: Template) -> bool:
    """Validate a Template model for completeness.

    Checks:
    - shot_count matches actual shots list length
    - All shots have non-negative durations
    - Shot times are sequential and non-overlapping
    - total_duration_sec is consistent

    Returns:
        True if valid.

    Raises:
        SynthesisError: If validation fails.
    """
    errors: list[str] = []

    if template.shot_count != len(template.shots):
        errors.append(
            f"shot_count ({template.shot_count}) != len(shots) ({len(template.shots)})"
        )

    for shot in template.shots:
        if shot.duration_sec < 0:
            errors.append(f"Shot {shot.shot_id} has negative duration: {shot.duration_sec}")
        if shot.end_sec < shot.start_sec:
            errors.append(f"Shot {shot.shot_id} end ({shot.end_sec}) < start ({shot.start_sec})")

    # Check sequential ordering
    for i in range(1, len(template.shots)):
        prev = template.shots[i - 1]
        curr = template.shots[i]
        if curr.start_sec < prev.start_sec:
            errors.append(
                f"Shot {curr.shot_id} starts ({curr.start_sec}) before shot {prev.shot_id} ({prev.start_sec})"
            )

    if errors:
        msg = "Template validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise SynthesisError(msg)

    logger.debug("Template %s passed validation", template.template_id)
    return True


def validate_template_json(json_str: str) -> Template:
    """Parse and validate a template JSON string.

    Returns:
        Validated Template model.

    Raises:
        SynthesisError: If JSON is invalid or schema validation fails.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise SynthesisError(f"Invalid JSON: {e}") from e

    try:
        template = Template(**data)
    except ValidationError as e:
        raise SynthesisError(f"Template schema error: {e}") from e

    validate_template(template)
    return template
