"""Self-learning taxonomy store for place classification.

Stores labels with bilingual text prompts and reference CLIP embeddings.
New labels are auto-discovered when shots can't be confidently classified.
Reference embeddings improve matching accuracy over time via EMA updates.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_LABELS: dict[str, dict[str, Any]] = {
    "exterior": {
        "text_prompts": [
            "a photo of a building exterior",
            "a hotel or resort exterior view",
            "건물 외관 사진",
        ],
    },
    "lobby": {
        "text_prompts": [
            "a photo of a hotel lobby",
            "a reception or entrance hall",
            "호텔 로비 사진",
        ],
    },
    "bedroom": {
        "text_prompts": [
            "a photo of a bedroom",
            "a hotel room with a bed",
            "침실 사진",
        ],
    },
    "bathroom": {
        "text_prompts": [
            "a photo of a bathroom",
            "a bathroom with a bathtub or shower",
            "욕실 사진",
        ],
    },
    "pool": {
        "text_prompts": [
            "a photo of a swimming pool",
            "a pool area at a resort",
            "수영장 사진",
        ],
    },
    "restaurant": {
        "text_prompts": [
            "a photo of a restaurant or dining area",
            "a cafe or bar interior",
            "레스토랑 또는 식당 사진",
        ],
    },
    "view": {
        "text_prompts": [
            "a scenic view or landscape photo",
            "a window view or balcony view",
            "전망 또는 풍경 사진",
        ],
    },
    "amenity": {
        "text_prompts": [
            "a photo of hotel amenities",
            "spa or gym or fitness center",
            "부대시설 사진",
        ],
    },
    "outdoor_bath": {
        "text_prompts": [
            "a photo of an outdoor hot spring bath",
            "an open-air onsen or hot tub",
            "야외 노천탕 사진",
            "온천 노천탕",
        ],
    },
    "kitchen": {
        "text_prompts": [
            "a photo of a kitchen",
            "a kitchenette in a hotel room",
            "주방 사진",
        ],
    },
    "garden": {
        "text_prompts": [
            "a photo of a garden or courtyard",
            "an outdoor garden area",
            "정원 사진",
        ],
    },
    "terrace": {
        "text_prompts": [
            "a photo of a terrace or balcony",
            "an outdoor deck with seating",
            "테라스 또는 발코니 사진",
        ],
    },
    "other": {
        "text_prompts": [
            "a miscellaneous photo",
            "기타 사진",
        ],
    },
}


class TaxonomyStore:
    """Persistent, self-learning taxonomy for place classification.

    Structure:
        data/taxonomy.json          - Labels + text prompts + metadata
        data/taxonomy.embeddings.npz - Reference CLIP image embeddings per label
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._embeddings_path = self.path.with_suffix(".embeddings.npz")
        self._data: dict[str, Any] = self._load()
        self._ref_embeddings: dict[str, np.ndarray] = self._load_embeddings()
        self._dirty = False

    # ── Persistence ──

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load taxonomy %s: %s", self.path, e)
        return self._create_default()

    def _create_default(self) -> dict[str, Any]:
        labels: dict[str, Any] = {}
        for name, info in _DEFAULT_LABELS.items():
            labels[name] = {
                "text_prompts": info["text_prompts"],
                "count": 0,
                "source": "default",
            }
        data = {"version": 1, "labels": labels}
        self._save_data(data)
        return data

    def _save_data(self, data: dict[str, Any] | None = None) -> None:
        if data is None:
            data = self._data
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_embeddings(self) -> dict[str, np.ndarray]:
        if self._embeddings_path.exists():
            try:
                data = np.load(str(self._embeddings_path), allow_pickle=False)
                return {k: data[k] for k in data.files}
            except Exception as e:
                logger.warning("Failed to load taxonomy embeddings: %s", e)
        return {}

    def _save_embeddings(self) -> None:
        if self._ref_embeddings:
            self._embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(self._embeddings_path), **self._ref_embeddings)

    def save(self) -> None:
        """Explicitly save all pending changes."""
        self._save_data()
        self._save_embeddings()
        self._dirty = False

    def flush(self) -> None:
        """Save only if there are pending changes."""
        if self._dirty:
            self.save()

    # ── Label access ──

    @property
    def labels(self) -> list[str]:
        return list(self._data.get("labels", {}).keys())

    @property
    def label_count(self) -> int:
        return len(self._data.get("labels", {}))

    def has_label(self, label: str) -> bool:
        return label in self._data.get("labels", {})

    def get_label_prompts(self) -> dict[str, list[str]]:
        """Return {label: [text_prompts]} for all labels."""
        result: dict[str, list[str]] = {}
        for label, info in self._data.get("labels", {}).items():
            result[label] = info.get("text_prompts", [f"a photo of a {label}"])
        return result

    def get_prompts_for(self, label: str) -> list[str]:
        info = self._data.get("labels", {}).get(label, {})
        return info.get("text_prompts", [f"a photo of a {label}"])

    # ── Label mutation ──

    def add_label(
        self,
        label: str,
        text_prompts: list[str] | None = None,
        reference_embedding: np.ndarray | None = None,
        source: str = "auto",
    ) -> bool:
        """Add a new label. Returns True if added, False if already exists."""
        if self.has_label(label):
            return False

        if text_prompts is None:
            text_prompts = _generate_default_prompts(label)

        self._data.setdefault("labels", {})[label] = {
            "text_prompts": text_prompts,
            "count": 0,
            "source": source,
            "added_at": datetime.now(timezone.utc).isoformat(),
        }

        if reference_embedding is not None:
            norm = np.linalg.norm(reference_embedding)
            if norm > 0:
                self._ref_embeddings[label] = reference_embedding / norm

        self.save()
        logger.info("Taxonomy: new label '%s' added (source: %s)", label, source)
        return True

    def update_reference(self, label: str, embedding: np.ndarray) -> None:
        """Update reference embedding with exponential moving average (alpha=0.1)."""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return

        normalized = embedding / norm
        existing = self._ref_embeddings.get(label)

        if existing is not None:
            alpha = 0.1
            updated = (1 - alpha) * existing + alpha * normalized
            updated_norm = np.linalg.norm(updated)
            if updated_norm > 0:
                self._ref_embeddings[label] = updated / updated_norm
        else:
            self._ref_embeddings[label] = normalized

        self._dirty = True

    def increment_count(self, label: str) -> None:
        info = self._data.get("labels", {}).get(label)
        if info:
            info["count"] = info.get("count", 0) + 1
            self._dirty = True

    # ── Embedding-based lookup ──

    def find_by_embedding(
        self, embedding: np.ndarray, threshold: float = 0.75,
    ) -> tuple[str | None, float]:
        """Find best matching label by cosine similarity to reference embeddings.

        Returns:
            (label, similarity) if above threshold, else (None, best_similarity).
        """
        if not self._ref_embeddings:
            return None, 0.0

        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None, 0.0

        emb_norm = embedding / norm
        best_label: str | None = None
        best_sim = 0.0

        for label, ref_emb in self._ref_embeddings.items():
            sim = float(np.dot(emb_norm, ref_emb))
            if sim > best_sim:
                best_sim = sim
                best_label = label

        if best_sim >= threshold:
            return best_label, best_sim
        return None, best_sim

    # ── Stats ──

    def get_stats(self) -> dict[str, Any]:
        labels = self._data.get("labels", {})
        return {
            "total_labels": len(labels),
            "auto_labels": sum(
                1 for v in labels.values() if v.get("source", "").startswith("auto")
            ),
            "default_labels": sum(
                1 for v in labels.values() if v.get("source") == "default"
            ),
            "labels_with_embeddings": len(self._ref_embeddings),
            "total_classifications": sum(v.get("count", 0) for v in labels.values()),
        }


def _generate_default_prompts(label: str) -> list[str]:
    """Generate bilingual text prompts for a label name."""
    return [
        f"a photo of {label}",
        f"a {label} area or room",
        f"{label} 사진",
    ]
