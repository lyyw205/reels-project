"""Tests for reels.analysis.taxonomy module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reels.analysis.taxonomy import TaxonomyStore, _generate_default_prompts


class TestTaxonomyStore:
    def test_creates_default_on_first_load(self, tmp_path: Path) -> None:
        store = TaxonomyStore(tmp_path / "taxonomy.json")
        assert store.label_count > 0
        assert "bedroom" in store.labels
        assert "bathroom" in store.labels
        assert "outdoor_bath" in store.labels
        assert store.path.exists()

    def test_loads_existing(self, tmp_path: Path) -> None:
        store1 = TaxonomyStore(tmp_path / "taxonomy.json")
        store1.add_label("test_label", text_prompts=["a test"])
        store1.save()

        store2 = TaxonomyStore(tmp_path / "taxonomy.json")
        assert "test_label" in store2.labels

    def test_has_label(self, tmp_path: Path) -> None:
        store = TaxonomyStore(tmp_path / "taxonomy.json")
        assert store.has_label("bedroom") is True
        assert store.has_label("nonexistent") is False

    def test_get_label_prompts(self, tmp_path: Path) -> None:
        store = TaxonomyStore(tmp_path / "taxonomy.json")
        prompts = store.get_label_prompts()
        assert "bedroom" in prompts
        assert len(prompts["bedroom"]) >= 2  # Bilingual

    def test_get_prompts_for(self, tmp_path: Path) -> None:
        store = TaxonomyStore(tmp_path / "taxonomy.json")
        prompts = store.get_prompts_for("bedroom")
        assert len(prompts) >= 2
        assert any("bedroom" in p for p in prompts)

    def test_add_label(self, tmp_path: Path) -> None:
        store = TaxonomyStore(tmp_path / "taxonomy.json")
        initial_count = store.label_count
        added = store.add_label("새로운장소", text_prompts=["a new place", "새 장소 사진"])
        assert added is True
        assert store.label_count == initial_count + 1
        assert store.has_label("새로운장소")

    def test_add_label_duplicate(self, tmp_path: Path) -> None:
        store = TaxonomyStore(tmp_path / "taxonomy.json")
        added = store.add_label("bedroom")
        assert added is False

    def test_add_label_with_embedding(self, tmp_path: Path) -> None:
        store = TaxonomyStore(tmp_path / "taxonomy.json")
        emb = np.random.randn(512).astype(np.float32)
        store.add_label("test_emb", reference_embedding=emb)

        # Reload and check
        store2 = TaxonomyStore(tmp_path / "taxonomy.json")
        ref_label, sim = store2.find_by_embedding(emb, threshold=0.9)
        assert ref_label == "test_emb"
        assert sim > 0.9

    def test_add_label_auto_prompts(self, tmp_path: Path) -> None:
        store = TaxonomyStore(tmp_path / "taxonomy.json")
        store.add_label("야외노천탕")
        prompts = store.get_prompts_for("야외노천탕")
        assert len(prompts) == 3
        assert any("야외노천탕" in p for p in prompts)

    def test_increment_count(self, tmp_path: Path) -> None:
        store = TaxonomyStore(tmp_path / "taxonomy.json")
        store.increment_count("bedroom")
        store.increment_count("bedroom")
        store.save()

        store2 = TaxonomyStore(tmp_path / "taxonomy.json")
        info = store2._data["labels"]["bedroom"]
        assert info["count"] == 2

    def test_update_reference_new(self, tmp_path: Path) -> None:
        store = TaxonomyStore(tmp_path / "taxonomy.json")
        emb = np.random.randn(512).astype(np.float32)
        store.update_reference("bedroom", emb)
        assert "bedroom" in store._ref_embeddings

    def test_update_reference_ema(self, tmp_path: Path) -> None:
        store = TaxonomyStore(tmp_path / "taxonomy.json")
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        store.update_reference("test", emb1)
        ref1 = store._ref_embeddings["test"].copy()

        store.update_reference("test", emb2)
        ref2 = store._ref_embeddings["test"]

        # EMA should shift toward emb2 but still be closer to emb1
        assert not np.allclose(ref1, ref2)
        assert np.dot(ref2, emb1 / np.linalg.norm(emb1)) > np.dot(ref2, emb2 / np.linalg.norm(emb2))

    def test_find_by_embedding_empty(self, tmp_path: Path) -> None:
        store = TaxonomyStore(tmp_path / "taxonomy.json")
        emb = np.random.randn(512).astype(np.float32)
        label, sim = store.find_by_embedding(emb)
        assert label is None
        assert sim == 0.0

    def test_find_by_embedding_match(self, tmp_path: Path) -> None:
        store = TaxonomyStore(tmp_path / "taxonomy.json")
        emb = np.random.randn(512).astype(np.float32)
        store.update_reference("pool", emb)

        label, sim = store.find_by_embedding(emb, threshold=0.9)
        assert label == "pool"
        assert sim > 0.9

    def test_find_by_embedding_below_threshold(self, tmp_path: Path) -> None:
        store = TaxonomyStore(tmp_path / "taxonomy.json")
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        store.update_reference("pool", emb1)

        emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        label, sim = store.find_by_embedding(emb2, threshold=0.9)
        assert label is None

    def test_find_by_embedding_zero_vector(self, tmp_path: Path) -> None:
        store = TaxonomyStore(tmp_path / "taxonomy.json")
        label, sim = store.find_by_embedding(np.zeros(512))
        assert label is None

    def test_flush_saves_when_dirty(self, tmp_path: Path) -> None:
        store = TaxonomyStore(tmp_path / "taxonomy.json")
        store.increment_count("bedroom")
        assert store._dirty is True
        store.flush()
        assert store._dirty is False

    def test_flush_noop_when_clean(self, tmp_path: Path) -> None:
        store = TaxonomyStore(tmp_path / "taxonomy.json")
        store.flush()  # Should not raise

    def test_get_stats(self, tmp_path: Path) -> None:
        store = TaxonomyStore(tmp_path / "taxonomy.json")
        emb = np.random.randn(512).astype(np.float32)
        store.add_label("auto_label", reference_embedding=emb, source="auto:test")

        stats = store.get_stats()
        assert stats["total_labels"] > 0
        assert stats["auto_labels"] == 1
        assert stats["labels_with_embeddings"] >= 1

    def test_corrupt_json_handled(self, tmp_path: Path) -> None:
        path = tmp_path / "taxonomy.json"
        path.write_text("{bad json!!!")
        store = TaxonomyStore(path)
        # Should fall back to defaults
        assert store.label_count > 0

    def test_embeddings_persist(self, tmp_path: Path) -> None:
        store1 = TaxonomyStore(tmp_path / "taxonomy.json")
        emb = np.random.randn(512).astype(np.float32)
        store1.update_reference("bedroom", emb)
        store1.save()

        store2 = TaxonomyStore(tmp_path / "taxonomy.json")
        assert "bedroom" in store2._ref_embeddings
        # Cosine similarity should be high
        sim = float(np.dot(
            store2._ref_embeddings["bedroom"],
            emb / np.linalg.norm(emb),
        ))
        assert sim > 0.99


class TestGenerateDefaultPrompts:
    def test_generates_bilingual(self) -> None:
        prompts = _generate_default_prompts("야외노천탕")
        assert len(prompts) == 3
        assert any("photo" in p for p in prompts)
        assert any("사진" in p for p in prompts)
