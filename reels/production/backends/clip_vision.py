"""CLIP-based local VLM backend for accommodation image analysis.

Uses the same CLIP model as the analysis pipeline (reels/analysis/place.py)
to extract accommodation features without requiring an external API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Pre-defined accommodation feature labels for CLIP text-image matching.
# Each entry has: Korean tag, English tag, category, and multiple English prompts
# for robust CLIP matching (max across prompts is taken as the score).
FEATURE_LABELS: list[dict[str, Any]] = [
    # ── amenity ──
    {
        "tag": "노천탕",
        "tag_en": "outdoor_bath",
        "category": "amenity",
        "prompts": ["outdoor bath", "open-air bath", "japanese onsen", "round stone bath outdoors"],
    },
    {
        "tag": "레인샤워",
        "tag_en": "rain_shower",
        "category": "amenity",
        "prompts": ["rain shower head", "rainfall shower", "modern shower room"],
    },
    {
        "tag": "자쿠지",
        "tag_en": "jacuzzi",
        "category": "amenity",
        "prompts": ["jacuzzi", "hot tub", "whirlpool bath"],
    },
    {
        "tag": "수영장",
        "tag_en": "pool",
        "category": "amenity",
        "prompts": ["swimming pool", "infinity pool", "private pool"],
    },
    {
        "tag": "사우나",
        "tag_en": "sauna",
        "category": "amenity",
        "prompts": ["sauna room", "steam room", "wooden sauna"],
    },
    {
        "tag": "벽난로",
        "tag_en": "fireplace",
        "category": "amenity",
        "prompts": ["fireplace", "fire pit", "cozy fireplace in room"],
    },
    {
        "tag": "종이 랜턴",
        "tag_en": "paper_lantern",
        "category": "amenity",
        "prompts": ["paper lantern", "japanese lantern", "traditional lantern lighting"],
    },
    {
        "tag": "디스플레이",
        "tag_en": "display_alcove",
        "category": "amenity",
        "prompts": ["display shelf", "decorative alcove", "art display niche"],
    },
    {
        "tag": "테라스",
        "tag_en": "terrace",
        "category": "amenity",
        "prompts": ["terrace", "balcony", "outdoor deck with seating"],
    },
    {
        "tag": "정원",
        "tag_en": "garden",
        "category": "amenity",
        "prompts": ["garden", "japanese garden", "courtyard garden"],
    },
    {
        "tag": "바베큐",
        "tag_en": "bbq",
        "category": "amenity",
        "prompts": ["barbecue grill", "outdoor bbq area", "grilling station"],
    },
    # ── scene ──
    {
        "tag": "침실",
        "tag_en": "bedroom",
        "category": "scene",
        "prompts": ["hotel bedroom", "twin beds", "luxury bedroom with bedding"],
    },
    {
        "tag": "욕실",
        "tag_en": "bathroom",
        "category": "scene",
        "prompts": ["bathroom", "vanity sink", "modern bathroom interior"],
    },
    {
        "tag": "거실",
        "tag_en": "living_room",
        "category": "scene",
        "prompts": ["living room", "lounge area", "hotel suite living space"],
    },
    {
        "tag": "타타미",
        "tag_en": "tatami_room",
        "category": "scene",
        "prompts": ["tatami room", "japanese style room", "traditional japanese living room"],
    },
    {
        "tag": "로비",
        "tag_en": "lobby",
        "category": "scene",
        "prompts": ["hotel lobby", "reception area", "entrance hall"],
    },
    {
        "tag": "외관",
        "tag_en": "exterior",
        "category": "scene",
        "prompts": ["hotel exterior", "building facade", "accommodation exterior view"],
    },
    {
        "tag": "데스크",
        "tag_en": "desk_area",
        "category": "scene",
        "prompts": ["desk", "work area", "study desk in room"],
    },
    # ── view ──
    {
        "tag": "오션뷰",
        "tag_en": "ocean_view",
        "category": "view",
        "prompts": ["ocean view", "sea view from window", "beach view from room"],
    },
    {
        "tag": "마운틴뷰",
        "tag_en": "mountain_view",
        "category": "view",
        "prompts": ["mountain view", "mountain scenery from window", "hilltop view"],
    },
    {
        "tag": "시티뷰",
        "tag_en": "city_view",
        "category": "view",
        "prompts": ["city view", "cityscape from window", "urban skyline view"],
    },
    {
        "tag": "석양",
        "tag_en": "sunset_view",
        "category": "view",
        "prompts": ["sunset view", "golden hour view", "sunset over the sea"],
    },
    # ── dining ──
    {
        "tag": "조식",
        "tag_en": "breakfast",
        "category": "dining",
        "prompts": ["breakfast table", "morning meal spread", "hotel breakfast buffet"],
    },
    {
        "tag": "레스토랑",
        "tag_en": "restaurant",
        "category": "dining",
        "prompts": ["restaurant interior", "dining room", "fine dining table setting"],
    },
    {
        "tag": "카페",
        "tag_en": "cafe",
        "category": "dining",
        "prompts": ["cafe", "coffee bar", "lounge cafe area"],
    },
    {
        "tag": "바",
        "tag_en": "bar",
        "category": "dining",
        "prompts": ["bar counter", "cocktail bar", "hotel bar lounge"],
    },
    # ── activity ──
    {
        "tag": "키즈",
        "tag_en": "kids_area",
        "category": "activity",
        "prompts": ["kids play area", "children playground", "family activity room"],
    },
    {
        "tag": "반려견",
        "tag_en": "pet_friendly",
        "category": "activity",
        "prompts": ["pet friendly hotel", "dog in hotel room", "pet amenities"],
    },
    {
        "tag": "불멍",
        "tag_en": "campfire",
        "category": "activity",
        "prompts": ["campfire", "bonfire pit", "outdoor fire pit at night"],
    },
    {
        "tag": "스파",
        "tag_en": "spa",
        "category": "activity",
        "prompts": ["spa treatment room", "massage room", "wellness spa"],
    },
]

# Default similarity threshold — labels below this are filtered out.
_DEFAULT_THRESHOLD = 0.2


class CLIPVisionBackend:
    """Local CLIP-based backend for accommodation feature extraction.

    Uses CLIP text-image similarity to match pre-defined accommodation
    feature labels against input images. No external API calls required.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = (config or {}).get("production", {}).get("feature_extraction", {})
        self.model_name: str = cfg.get("clip_model", "openai/clip-vit-base-patch32")
        self.threshold: float = cfg.get("clip_threshold", _DEFAULT_THRESHOLD)
        self.max_features_per_image: int = cfg.get("max_features_per_image", 5)
        self._model: Any = None
        self._processor: Any = None
        self._text_embeddings: np.ndarray | None = None
        self._prompt_label_map: list[int] | None = None

    def _load_model(self) -> None:
        """Lazy-load CLIP model, processor, and pre-compute text embeddings."""
        if self._model is not None:
            return

        import torch
        from transformers import CLIPModel, CLIPProcessor

        logger.info("Loading CLIP model for production backend: %s", self.model_name)
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name)
        self._model.eval()

        # Pre-compute text embeddings for all feature label prompts
        all_prompts: list[str] = []
        prompt_to_label_idx: list[int] = []
        for idx, label in enumerate(FEATURE_LABELS):
            for prompt in label["prompts"]:
                all_prompts.append(prompt)
                prompt_to_label_idx.append(idx)

        inputs = self._processor(text=all_prompts, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = self._model.get_text_features(**inputs)
            # L2-normalize for cosine similarity
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self._text_embeddings = text_features.cpu().numpy()
        self._prompt_label_map = prompt_to_label_idx

    async def analyze_image(
        self, image: Path, prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Analyze single image and return feature dicts.

        The `prompt` parameter is accepted for VLMBackend compatibility
        but ignored — CLIP uses pre-defined feature labels instead.
        """
        import torch
        from PIL import Image

        self._load_model()
        assert self._model is not None
        assert self._processor is not None
        assert self._text_embeddings is not None
        assert self._prompt_label_map is not None

        # Load and encode image
        pil_image = Image.open(image).convert("RGB")
        inputs = self._processor(images=[pil_image], return_tensors="pt")

        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        img_emb = image_features.cpu().numpy()  # (1, dim)

        # Cosine similarity against all text prompts
        similarities = (img_emb @ self._text_embeddings.T).flatten()  # (num_prompts,)

        # Aggregate: max similarity per label
        label_scores: dict[int, float] = {}
        for prompt_idx, label_idx in enumerate(self._prompt_label_map):
            sim = float(similarities[prompt_idx])
            if label_idx not in label_scores or sim > label_scores[label_idx]:
                label_scores[label_idx] = sim

        # Filter by threshold and build result dicts
        results: list[dict[str, Any]] = []
        for label_idx, score in label_scores.items():
            if score < self.threshold:
                continue
            label = FEATURE_LABELS[label_idx]
            results.append({
                "tag": label["tag"],
                "tag_en": label["tag_en"],
                "category": label["category"],
                "confidence": round(float(score), 4),
                "description": f"CLIP similarity {score:.3f}",
            })

        # Sort by confidence descending, take top N
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[: self.max_features_per_image]

    async def analyze_batch(
        self, images: list[Path], prompt: str | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Analyze multiple images."""
        results = []
        for img in images:
            result = await self.analyze_image(img, prompt)
            results.append(result)
        return results

    def cleanup(self) -> None:
        """Release model resources."""
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._text_embeddings = None
            self._prompt_label_map = None
            logger.info("CLIPVisionBackend: model unloaded")
