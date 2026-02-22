"""Validate JSON output files against Pydantic models.

Used by OMC team agents to verify their output conforms to the expected schema
before handing off to the next agent.

CLI usage::

    python -m reels.production.omc_helpers.validate_output <type> <json_file>

Where ``type`` is one of: brief, shot_plan, copies, storyboard, qa_report.

Exit code 0 on success, 1 on validation failure (details printed to stderr).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError


_MODEL_MAP: dict[str, tuple[str, str]] = {
    "brief": (
        "reels.production.creative_team.models",
        "CreativeBrief",
    ),
    "shot_plan": (
        "reels.production.creative_team.models",
        "ShotPlan",
    ),
    "copies": (
        "reels.production.models",
        "ShotCopy",
    ),
    "storyboard": (
        "reels.production.models",
        "Storyboard",
    ),
    "qa_report": (
        "reels.production.creative_team.models",
        "QAReport",
    ),
}

VALID_TYPES = tuple(_MODEL_MAP.keys())


def _get_model_class(type_name: str) -> type:
    """Lazy-import and return the Pydantic model class for *type_name*."""
    if type_name not in _MODEL_MAP:
        raise ValueError(
            f"Unknown type '{type_name}'. Valid types: {', '.join(VALID_TYPES)}"
        )

    module_path, class_name = _MODEL_MAP[type_name]
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def validate_output(type_name: str, data: Any) -> list[str]:
    """Validate *data* against the Pydantic model for *type_name*.

    Args:
        type_name: One of ``brief``, ``shot_plan``, ``copies``,
            ``storyboard``, ``qa_report``.
        data: Parsed JSON data (dict or list).

    Returns:
        Empty list on success, or list of error description strings.
    """
    model_cls = _get_model_class(type_name)

    errors: list[str] = []

    if type_name == "copies":
        # copies is a list of ShotCopy
        if not isinstance(data, list):
            return [f"Expected a JSON array for '{type_name}', got {type(data).__name__}"]
        for i, item in enumerate(data):
            try:
                model_cls.model_validate(item)
            except ValidationError as e:
                for err in e.errors():
                    loc = " -> ".join(str(x) for x in err["loc"])
                    errors.append(f"copies[{i}].{loc}: {err['msg']}")
    else:
        try:
            model_cls.model_validate(data)
        except ValidationError as e:
            for err in e.errors():
                loc = " -> ".join(str(x) for x in err["loc"])
                errors.append(f"{loc}: {err['msg']}")

    return errors


def main() -> None:
    """CLI entry point: validate a JSON file against a model type."""
    if len(sys.argv) < 3:
        print(
            f"Usage: python -m reels.production.omc_helpers.validate_output "
            f"<{'|'.join(VALID_TYPES)}> <json_file>",
            file=sys.stderr,
        )
        sys.exit(1)

    type_name = sys.argv[1]
    json_path = Path(sys.argv[2])

    if type_name not in VALID_TYPES:
        print(
            f"Error: Unknown type '{type_name}'. Valid: {', '.join(VALID_TYPES)}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        data = json.loads(json_path.read_text())
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading {json_path}: {e}", file=sys.stderr)
        sys.exit(1)

    errors = validate_output(type_name, data)

    if errors:
        print(f"Validation FAILED for {type_name}:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"OK: {type_name} validation passed.")


if __name__ == "__main__":
    main()
