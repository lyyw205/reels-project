"""Content-hash based response cache for API calls."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ResponseCache:
    """Cache API responses keyed by content hash of inputs."""

    def __init__(self, cache_dir: str | Path = ".cache/vlm") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> list[dict[str, Any]] | None:
        """Get cached response. Returns None on miss."""
        path = self._key_path(key)
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self._hits += 1
                logger.debug("Cache hit: %s", key[:16])
                return data
            except (json.JSONDecodeError, OSError):
                return None
        self._misses += 1
        return None

    def put(self, key: str, value: list[dict[str, Any]]) -> None:
        """Store response in cache."""
        path = self._key_path(key)
        try:
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2))
        except OSError as e:
            logger.warning("Cache write failed: %s", e)

    def hash_file(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file contents."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def hash_text(self, text: str) -> str:
        """Compute SHA-256 hash of text."""
        return hashlib.sha256(text.encode()).hexdigest()

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses}

    def _key_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"
