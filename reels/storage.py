"""Template archiver for persistent storage.

Archives template JSON + keyframes to a durable location
(data/templates/{template_id}/) so they survive work directory cleanup
and are available for video production (Remotion rendering etc.).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from reels.models.template import Template

logger = logging.getLogger(__name__)


class TemplateArchiver:
    """Archive templates and keyframes to persistent storage."""

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)

    def archive(
        self,
        template: Template,
        work_dir: Path,
        source_video: str | None = None,
    ) -> Path:
        """Archive template + keyframes to persistent storage.

        Copies template JSON and keyframe images from work_dir to
        data/templates/{template_id}/. Returns the archive directory path.
        """
        archive_dir = self.base_dir / template.template_id
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Save template JSON
        template_path = archive_dir / "template.json"
        template_path.write_text(template.model_dump_json(indent=2))
        logger.info("Archived template: %s", template_path)

        # Copy keyframes
        keyframes_src = work_dir / "keyframes"
        keyframes_dst = archive_dir / "keyframes"
        if keyframes_src.exists():
            if keyframes_dst.exists():
                shutil.rmtree(keyframes_dst)
            shutil.copytree(keyframes_src, keyframes_dst)
            count = sum(1 for _ in keyframes_dst.iterdir())
            logger.info("Archived %d keyframes: %s", count, keyframes_dst)

        # Save source reference
        if source_video:
            meta_path = archive_dir / "source.txt"
            meta_path.write_text(source_video)

        return archive_dir

    def get(self, template_id: str) -> Path | None:
        """Get archive directory for a template. Returns None if not found."""
        path = self.base_dir / template_id
        return path if path.exists() else None

    def get_template(self, template_id: str) -> Template | None:
        """Load a template from the archive."""
        archive_dir = self.get(template_id)
        if archive_dir is None:
            return None

        template_path = archive_dir / "template.json"
        if not template_path.exists():
            return None

        return Template.model_validate_json(template_path.read_text())

    def get_keyframe_dir(self, template_id: str) -> Path | None:
        """Get keyframe directory for a template."""
        archive_dir = self.get(template_id)
        if archive_dir is None:
            return None

        kf_dir = archive_dir / "keyframes"
        return kf_dir if kf_dir.exists() else None

    def list_templates(self) -> list[str]:
        """List all archived template IDs."""
        if not self.base_dir.exists():
            return []
        return sorted(
            d.name for d in self.base_dir.iterdir()
            if d.is_dir() and (d / "template.json").exists()
        )

    def delete(self, template_id: str) -> bool:
        """Delete an archived template. Returns True if deleted."""
        archive_dir = self.get(template_id)
        if archive_dir is None:
            return False
        shutil.rmtree(archive_dir)
        logger.info("Deleted archive: %s", template_id)
        return True
