"""Speech recognition analyzer using faster-whisper."""

from __future__ import annotations

import logging
from pathlib import Path

from reels.analysis.base import AnalysisContext
from reels.models.analysis import SpeechResult, SpeechSegment, SpeechWord
from reels.models.shot import Shot

logger = logging.getLogger(__name__)


class SpeechAnalyzer:
    """Transcribe speech using faster-whisper with word-level timestamps."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("analysis", {}).get("speech", {})
        self.model_size: str = cfg.get("model", "small")
        self.language: str = cfg.get("language", "ko")
        self.compute_type: str = cfg.get("compute_type", "int8")
        self.word_timestamps: bool = cfg.get("word_timestamps", True)
        self.vad_filter: bool = cfg.get("vad_filter", True)
        self._model = None

    @property
    def name(self) -> str:
        return "speech"

    def _load_model(self) -> None:
        """Lazy-load faster-whisper model."""
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        logger.info("Loading Whisper model: %s (compute=%s)", self.model_size, self.compute_type)
        self._model = WhisperModel(
            self.model_size,
            device="cpu",
            compute_type=self.compute_type,
        )

    def analyze_shot(self, shot: Shot, context: AnalysisContext) -> SpeechResult:
        """Transcribe speech in a shot's time range."""
        if context.audio_path is None or not Path(context.audio_path).exists():
            return SpeechResult(has_speech=False)

        self._load_model()

        try:
            segments_gen, info = self._model.transcribe(
                str(context.audio_path),
                language=self.language,
                word_timestamps=self.word_timestamps,
                vad_filter=self.vad_filter,
            )

            segments: list[SpeechSegment] = []
            has_speech = False

            for segment in segments_gen:
                # Only include segments that overlap with this shot
                if segment.end < shot.start_sec or segment.start > shot.end_sec:
                    continue

                has_speech = True
                words = []
                if segment.words:
                    for w in segment.words:
                        if w.end >= shot.start_sec and w.start <= shot.end_sec:
                            words.append(SpeechWord(
                                word=w.word.strip(),
                                start=round(w.start, 3),
                                end=round(w.end, 3),
                                probability=round(w.probability, 3),
                            ))

                segments.append(SpeechSegment(
                    text=segment.text.strip(),
                    start=round(max(segment.start, shot.start_sec), 3),
                    end=round(min(segment.end, shot.end_sec), 3),
                    words=words,
                ))

            return SpeechResult(has_speech=has_speech, segments=segments)

        except Exception as e:
            logger.warning("Speech analysis failed for shot %d: %s", shot.shot_id, e)
            return SpeechResult(has_speech=False)

    def analyze_batch(self, shots: list[Shot], context: AnalysisContext) -> list[SpeechResult]:
        return [self.analyze_shot(shot, context) for shot in shots]

    def cleanup(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("SpeechAnalyzer: model unloaded")
