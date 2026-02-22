"""SQLite template repository with JSON1 extension support."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from reels.exceptions import DatabaseError
from reels.models.template import Template

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS templates (
    template_id TEXT PRIMARY KEY,
    source_url TEXT,
    aspect TEXT NOT NULL DEFAULT '9:16',
    fps INTEGER NOT NULL DEFAULT 30,
    total_duration_sec REAL NOT NULL,
    shot_count INTEGER NOT NULL,
    template_json TEXT NOT NULL,
    avg_bpm REAL,
    dominant_place TEXT,
    has_speech BOOLEAN DEFAULT FALSE,
    camera_types TEXT,
    archive_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_templates_place ON templates(dominant_place)",
    "CREATE INDEX IF NOT EXISTS idx_templates_bpm ON templates(avg_bpm)",
    "CREATE INDEX IF NOT EXISTS idx_templates_duration ON templates(total_duration_sec)",
]


class TemplateRepository:
    """SQLite-backed template storage with JSON search capabilities."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        try:
            conn.execute(_CREATE_TABLE)
            for idx_sql in _CREATE_INDEXES:
                conn.execute(idx_sql)
            conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to initialize database: {e}") from e

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def save(self, template: Template, archive_path: str | None = None) -> None:
        """Save a template to the database.

        Args:
            template: Template model to save.
            archive_path: Path to the archived template directory.

        Raises:
            DatabaseError: If save fails.
        """
        conn = self._get_conn()

        # Extract searchable fields
        avg_bpm = self._extract_avg_bpm(template)
        dominant_place = self._extract_dominant_place(template)
        has_speech = any(s.audio.has_speech for s in template.shots)
        camera_types = json.dumps(list(set(str(s.camera.type) for s in template.shots)))

        try:
            conn.execute(
                """INSERT OR REPLACE INTO templates
                   (template_id, source_url, aspect, fps, total_duration_sec,
                    shot_count, template_json, avg_bpm, dominant_place,
                    has_speech, camera_types, archive_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    template.template_id,
                    template.source_url,
                    template.format.aspect,
                    template.format.fps,
                    template.total_duration_sec,
                    template.shot_count,
                    template.model_dump_json(),
                    avg_bpm,
                    dominant_place,
                    has_speech,
                    camera_types,
                    archive_path,
                ),
            )
            conn.commit()
            logger.info("Saved template: %s", template.template_id)
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to save template {template.template_id}: {e}") from e

    def get(self, template_id: str) -> Template | None:
        """Retrieve a template by ID.

        Returns:
            Template model or None if not found.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT template_json FROM templates WHERE template_id = ?",
            (template_id,),
        ).fetchone()

        if row is None:
            return None

        return Template.model_validate_json(row["template_json"])

    def list_all(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """List templates with summary info."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT template_id, source_url, aspect, total_duration_sec,
                      shot_count, dominant_place, avg_bpm, has_speech,
                      archive_path, created_at
               FROM templates ORDER BY created_at DESC LIMIT ? OFFSET ?""",
            (limit, offset),
        ).fetchall()

        return [dict(row) for row in rows]

    def search_by_place(self, place: str) -> list[Template]:
        """Search templates by dominant place label."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT template_json FROM templates WHERE dominant_place = ?",
            (place,),
        ).fetchall()

        return [Template.model_validate_json(row["template_json"]) for row in rows]

    def search_by_camera_type(self, camera_type: str) -> list[Template]:
        """Search templates containing a specific camera type (uses json_each)."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT DISTINCT t.template_json
               FROM templates t, json_each(t.camera_types) j
               WHERE j.value = ?""",
            (camera_type,),
        ).fetchall()

        return [Template.model_validate_json(row["template_json"]) for row in rows]

    def search_by_duration(
        self, min_sec: float = 0, max_sec: float = 999,
    ) -> list[Template]:
        """Search templates by duration range."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT template_json FROM templates WHERE total_duration_sec BETWEEN ? AND ?",
            (min_sec, max_sec),
        ).fetchall()

        return [Template.model_validate_json(row["template_json"]) for row in rows]

    def search_composite(
        self,
        places: list[str] | None = None,
        duration_range: tuple[float, float] | None = None,
        has_speech: bool | None = None,
        bpm_range: tuple[float, float] | None = None,
        camera_types: list[str] | None = None,
        limit: int = 10,
    ) -> list[Template]:
        """Search templates with multiple filters combined."""
        conditions: list[str] = []
        params: list[Any] = []

        if places:
            placeholders = ", ".join("?" * len(places))
            conditions.append(f"t.dominant_place IN ({placeholders})")
            params.extend(places)

        if duration_range is not None:
            conditions.append("t.total_duration_sec BETWEEN ? AND ?")
            params.extend([duration_range[0], duration_range[1]])

        if has_speech is not None:
            conditions.append("t.has_speech = ?")
            params.append(has_speech)

        if bpm_range is not None:
            conditions.append("t.avg_bpm BETWEEN ? AND ?")
            params.extend([bpm_range[0], bpm_range[1]])

        if camera_types:
            placeholders = ", ".join("?" * len(camera_types))
            conditions.append(f"j.value IN ({placeholders})")
            params.extend(camera_types)

        if camera_types:
            from_clause = "FROM templates t, json_each(t.camera_types) j"
            select_clause = "SELECT DISTINCT t.template_json"
        else:
            from_clause = "FROM templates t"
            select_clause = "SELECT t.template_json"

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"{select_clause} {from_clause} {where_clause} ORDER BY t.created_at DESC LIMIT ?"
        params.append(limit)

        conn = self._get_conn()
        rows = conn.execute(sql, params).fetchall()
        return [Template.model_validate_json(row["template_json"]) for row in rows]

    def delete(self, template_id: str) -> bool:
        """Delete a template by ID. Returns True if deleted."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM templates WHERE template_id = ?",
            (template_id,),
        )
        conn.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Count total templates in database."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM templates").fetchone()
        return row["cnt"]

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @staticmethod
    def _extract_avg_bpm(template: Template) -> float | None:
        """Extract average BPM from template shots' rhythm data."""
        bpms = [s.bpm for s in template.shots if s.bpm is not None and s.bpm > 0]
        if not bpms:
            return None
        return round(sum(bpms) / len(bpms), 1)

    @staticmethod
    def _extract_dominant_place(template: Template) -> str | None:
        """Find the most common place label across shots."""
        if not template.shots:
            return None
        place_counts: dict[str, int] = {}
        for shot in template.shots:
            label = shot.place_label
            place_counts[label] = place_counts.get(label, 0) + 1
        return max(place_counts, key=place_counts.get)  # type: ignore[arg-type]
