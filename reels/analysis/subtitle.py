"""Subtitle/text overlay analyzer using OCR (EasyOCR default, PaddleOCR optional)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from reels.analysis.base import AnalysisContext
from reels.models.analysis import SubtitleEntry, SubtitleResult
from reels.models.shot import Shot

logger = logging.getLogger(__name__)


class SubtitleAnalyzer:
    """Extract hardcoded subtitles/text overlays using OCR with majority voting."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("analysis", {}).get("subtitle", {})
        self.engine: str = cfg.get("engine", "easyocr")
        self.languages: list[str] = cfg.get("languages", ["ko", "en"])
        self.confidence_threshold: float = cfg.get("confidence_threshold", 0.6)
        self.voting_window: int = cfg.get("voting_window_frames", 5)
        self.exclude_bottom_ratio: float = cfg.get("exclude_bottom_ratio", 0.85)
        self.dedup_iou: float = cfg.get("dedup_iou_threshold", 0.5)
        self._reader = None

    @property
    def name(self) -> str:
        return "subtitle"

    def _load_reader(self) -> None:
        """Lazy-load OCR engine."""
        if self._reader is not None:
            return

        if self.engine == "easyocr":
            import easyocr
            logger.info("Loading EasyOCR: languages=%s", self.languages)
            self._reader = easyocr.Reader(self.languages, gpu=False)
        elif self.engine == "paddleocr":
            try:
                from paddleocr import PaddleOCR
                lang = self.languages[0] if self.languages else "ko"
                self._reader = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
            except ImportError:
                logger.warning("PaddleOCR not installed, falling back to EasyOCR")
                import easyocr
                self._reader = easyocr.Reader(self.languages, gpu=False)
                self.engine = "easyocr"

    def analyze_shot(self, shot: Shot, context: AnalysisContext) -> SubtitleResult:
        """Extract text overlays from a shot using sampled frames + majority voting."""
        self._load_reader()

        cap = cv2.VideoCapture(str(context.video_path))
        if not cap.isOpened():
            return SubtitleResult()

        fps = context.metadata.fps
        total_frames = shot.end_frame - shot.start_frame

        # Sample frames at regular intervals
        sample_count = max(1, min(self.voting_window, total_frames // max(1, int(fps * 0.5))))
        step = max(1, total_frames // (sample_count + 1))
        sample_frames = [shot.start_frame + step * (i + 1) for i in range(sample_count)]

        all_detections: list[list[dict[str, Any]]] = []

        for frame_num in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            detections = self._ocr_frame(frame)
            all_detections.append(detections)

        cap.release()

        # Majority voting across frames
        entries = self._majority_vote(all_detections, shot)
        return SubtitleResult(texts=entries)

    def analyze_batch(self, shots: list[Shot], context: AnalysisContext) -> list[SubtitleResult]:
        return [self.analyze_shot(shot, context) for shot in shots]

    def cleanup(self) -> None:
        if self._reader is not None:
            del self._reader
            self._reader = None
            logger.info("SubtitleAnalyzer: reader unloaded")

    def _ocr_frame(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Run OCR on a single frame, excluding bottom UI area."""
        h, w = frame.shape[:2]

        # Exclude bottom UI area (preserved from old code: _EXCLUDE_Y_THRESHOLD = 0.85)
        crop_h = int(h * self.exclude_bottom_ratio)
        roi = frame[:crop_h]

        detections = []

        if self.engine == "easyocr":
            results = self._reader.readtext(roi)
            for bbox, text, conf in results:
                if conf < self.confidence_threshold:
                    continue
                # Normalize bbox to 0-1 coordinates
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                box = {
                    "x": round(min(xs) / w, 4),
                    "y": round(min(ys) / h, 4),
                    "w": round((max(xs) - min(xs)) / w, 4),
                    "h": round((max(ys) - min(ys)) / h, 4),
                }
                detections.append({"text": text.strip(), "box": box, "confidence": float(conf)})

        return detections

    def _majority_vote(
        self,
        all_detections: list[list[dict[str, Any]]],
        shot: Shot,
    ) -> list[SubtitleEntry]:
        """Aggregate OCR results across frames with majority voting.

        Preserved from old code: on tie, choose longest text.
        """
        if not all_detections:
            return []

        text_counts: dict[str, int] = {}
        text_info: dict[str, dict[str, Any]] = {}

        for detections in all_detections:
            for det in detections:
                text = det["text"]
                text_counts[text] = text_counts.get(text, 0) + 1
                # Keep best confidence version
                if text not in text_info or det["confidence"] > text_info[text].get("confidence", 0):
                    text_info[text] = det

        # Filter: must appear in at least half of sampled frames
        min_votes = max(1, len(all_detections) // 2)
        entries = []

        for text, count in text_counts.items():
            if count >= min_votes:
                info = text_info[text]
                entries.append(SubtitleEntry(
                    text=text,
                    box=info.get("box", {}),
                    style={},
                    start_sec=shot.start_sec,
                    end_sec=shot.end_sec,
                    confidence=info.get("confidence", 0.0),
                ))

        # Sort by confidence descending; on tie, longest text first
        entries.sort(key=lambda e: (-e.confidence, -len(e.text)))
        return entries
