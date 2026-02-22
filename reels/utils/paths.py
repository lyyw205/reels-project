"""Path management utilities."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def work_subdir(work_dir: Path, name: str) -> Path:
    """Create and return a named subdirectory under work_dir."""
    return ensure_dir(work_dir / name)
