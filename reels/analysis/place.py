"""Place/subject analyzer using CLIP embeddings with adaptive taxonomy."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from reels.analysis.base import AnalysisContext
from reels.analysis.taxonomy import TaxonomyStore
from reels.models.analysis import PlaceResult
from reels.models.shot import Shot

logger = logging.getLogger(__name__)


class PlaceAnalyzer:
    """Classify shot location/subject using CLIP + self-learning taxonomy.

    3-stage classification:
        1. CLIP text matching against taxonomy labels (bilingual prompts)
        2. Reference image embedding matching (cosine similarity)
        3. Expanded candidate discovery from filename hint
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("analysis", {}).get("place", {})
        self.model_name: str = cfg.get("model", "openai/clip-vit-base-patch32")
        self.batch_size: int = cfg.get("batch_size", 16)

        # Taxonomy settings
        self.taxonomy_path = Path(cfg.get("taxonomy_path", "data/taxonomy.json"))
        self.auto_discover: bool = cfg.get("auto_discover", True)
        self.discovery_threshold: float = cfg.get("discovery_threshold", 0.15)
        self.ref_match_threshold: float = cfg.get("ref_match_threshold", 0.75)

        self._model = None
        self._processor = None
        self._taxonomy_store: TaxonomyStore | None = None

    @property
    def name(self) -> str:
        return "place"

    @property
    def taxonomy_store(self) -> TaxonomyStore:
        if self._taxonomy_store is None:
            self._taxonomy_store = TaxonomyStore(self.taxonomy_path)
        return self._taxonomy_store

    def _load_model(self) -> None:
        """Lazy-load CLIP model and processor."""
        if self._model is not None:
            return

        from transformers import CLIPModel, CLIPProcessor

        logger.info("Loading CLIP model: %s", self.model_name)
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name)
        self._model.eval()

    def analyze_shot(self, shot: Shot, context: AnalysisContext) -> PlaceResult:
        """Classify a single shot by its keyframes."""
        self._load_model()

        images = self._load_keyframes(shot, context)
        if not images:
            return PlaceResult(place_label="other", confidence=0.0)

        source_hint = context.metadata.source or ""
        return self._classify_with_discovery(images, source_hint)

    def analyze_batch(self, shots: list[Shot], context: AnalysisContext) -> list[PlaceResult]:
        """Classify multiple shots."""
        return [self.analyze_shot(shot, context) for shot in shots]

    def cleanup(self) -> None:
        """Unload CLIP model and flush taxonomy changes."""
        if self._taxonomy_store is not None:
            self._taxonomy_store.flush()

        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            logger.info("PlaceAnalyzer: model unloaded")

    # ── Classification pipeline ──

    def _classify_with_discovery(
        self, images: list[np.ndarray], source_hint: str,
    ) -> PlaceResult:
        """3-stage classification with auto-discovery."""
        store = self.taxonomy_store
        label_prompts = store.get_label_prompts()

        # Stage 1: CLIP text classification against taxonomy
        text_result = self._clip_classify(images, label_prompts)

        # Get image embedding for reference matching
        img_embedding = self._get_image_embedding(images[0])

        if text_result.confidence >= self.discovery_threshold:
            # Confident match - update reference and return
            store.update_reference(text_result.place_label, img_embedding)
            store.increment_count(text_result.place_label)
            return text_result

        # Stage 2: Check reference image embeddings
        ref_label, ref_sim = store.find_by_embedding(
            img_embedding, self.ref_match_threshold,
        )
        if ref_label:
            store.update_reference(ref_label, img_embedding)
            store.increment_count(ref_label)
            return PlaceResult(
                place_label=ref_label,
                confidence=ref_sim,
                top_labels=text_result.top_labels,
            )

        # Stage 3: Try expanded candidates from filename hint
        if self.auto_discover and source_hint:
            expanded = self._expand_candidates(source_hint)
            if expanded:
                all_prompts = dict(label_prompts)
                all_prompts.update(expanded)

                expanded_result = self._clip_classify(images, all_prompts)

                if (
                    expanded_result.place_label in expanded
                    and expanded_result.confidence > text_result.confidence
                ):
                    new_label = expanded_result.place_label
                    store.add_label(
                        new_label,
                        text_prompts=expanded[new_label],
                        reference_embedding=img_embedding,
                        source=f"auto:{Path(source_hint).stem}",
                    )
                    store.increment_count(new_label)
                    return expanded_result

        # Fallback: use best text result
        store.update_reference(text_result.place_label, img_embedding)
        store.increment_count(text_result.place_label)
        return text_result

    def _expand_candidates(self, source_hint: str) -> dict[str, list[str]]:
        """Generate expanded candidate labels from filename hint."""
        stem = Path(source_hint).stem

        if self.taxonomy_store.has_label(stem):
            return {}

        return {
            stem: [
                f"a photo of {stem}",
                f"{stem}",
                f"{stem} 사진",
            ],
        }

    # ── CLIP operations ──

    def _clip_classify(
        self, images: list[np.ndarray], label_prompts: dict[str, list[str]],
    ) -> PlaceResult:
        """Multi-prompt CLIP classification.

        For each label, takes the max score across its text prompts.
        This handles bilingual prompts effectively.
        """
        import torch
        from PIL import Image

        pil_images = [Image.fromarray(img) for img in images]

        # Flatten prompts with label tracking
        all_prompts: list[str] = []
        prompt_labels: list[str] = []
        for label, prompts in label_prompts.items():
            for p in prompts:
                all_prompts.append(p)
                prompt_labels.append(label)

        inputs = self._processor(
            text=all_prompts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_image.mean(dim=0)
            probs = logits.softmax(dim=-1).cpu().numpy()

        # Aggregate by label: max probability across prompts
        label_scores: dict[str, float] = {}
        for prob, label in zip(probs, prompt_labels):
            label_scores[label] = max(label_scores.get(label, 0.0), float(prob))

        sorted_labels = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
        top_labels = [(label, score) for label, score in sorted_labels[:5]]
        best_label, best_score = sorted_labels[0]

        return PlaceResult(
            place_label=best_label,
            confidence=best_score,
            top_labels=top_labels,
        )

    def _get_image_embedding(self, image: np.ndarray) -> np.ndarray:
        """Extract CLIP image embedding for reference matching."""
        import torch
        from PIL import Image

        pil_image = Image.fromarray(image)
        inputs = self._processor(images=[pil_image], return_tensors="pt")

        with torch.no_grad():
            features = self._model.get_image_features(
                pixel_values=inputs["pixel_values"],
            )

        return features.cpu().numpy().flatten()

    # ── Keyframe loading ──

    def _load_keyframes(self, shot: Shot, context: AnalysisContext) -> list[np.ndarray]:
        """Load keyframe images for a shot."""
        images = []
        keyframe_dir = Path(context.work_dir) / "keyframes"

        for kf_path in shot.keyframe_paths:
            full_path = keyframe_dir / kf_path
            if full_path.exists():
                img = cv2.imread(str(full_path))
                if img is not None:
                    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not images and context.video_path.exists():
            # Fallback: read middle frame
            cap = cv2.VideoCapture(str(context.video_path))
            mid_frame = (shot.start_frame + shot.end_frame) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            cap.release()
            if ret:
                images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        return images
