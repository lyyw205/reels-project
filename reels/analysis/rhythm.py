"""Rhythm/beat analyzer using librosa."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from reels.analysis.base import AnalysisContext
from reels.models.analysis import RhythmResult
from reels.models.shot import Shot

logger = logging.getLogger(__name__)


class RhythmAnalyzer:
    """Analyze beat/onset patterns using librosa for cut-on-beat detection."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("analysis", {}).get("rhythm", {})
        self.hop_length: int = cfg.get("hop_length", 512)
        self.onset_threshold: float = cfg.get("onset_strength_threshold", 0.5)
        self._audio: np.ndarray | None = None
        self._sr: int | float = 0
        self._beats: np.ndarray | None = None
        self._onsets: np.ndarray | None = None
        self._bpm: float = 0.0

    @property
    def name(self) -> str:
        return "rhythm"

    def _load_audio(self, context: AnalysisContext) -> bool:
        """Load and analyze full audio track (once per video)."""
        if self._audio is not None:
            return True

        if context.audio_path is None or not Path(context.audio_path).exists():
            return False

        import librosa

        logger.info("Loading audio for rhythm analysis: %s", context.audio_path)
        self._audio, self._sr = librosa.load(str(context.audio_path), sr=None)

        # Beat tracking
        tempo, beat_frames = librosa.beat.beat_track(
            y=self._audio, sr=self._sr, hop_length=self.hop_length,
        )
        self._bpm = float(np.atleast_1d(tempo)[0])
        self._beats = librosa.frames_to_time(beat_frames, sr=self._sr, hop_length=self.hop_length)

        # Onset detection
        onset_env = librosa.onset.onset_strength(
            y=self._audio, sr=self._sr, hop_length=self.hop_length,
        )
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=self._sr, hop_length=self.hop_length,
        )
        self._onsets = librosa.frames_to_time(onset_frames, sr=self._sr, hop_length=self.hop_length)

        logger.info("Rhythm analysis: BPM=%.1f, %d beats, %d onsets", self._bpm, len(self._beats), len(self._onsets))
        return True

    def analyze_shot(self, shot: Shot, context: AnalysisContext) -> RhythmResult:
        """Analyze rhythm within a shot's time range."""
        if not self._load_audio(context):
            return RhythmResult()

        # Find beats and onsets within shot boundaries
        beats_in_shot = [
            round(float(b), 3) for b in self._beats  # type: ignore[union-attr]
            if shot.start_sec <= b <= shot.end_sec
        ]
        onsets_in_shot = [
            round(float(o), 3) for o in self._onsets  # type: ignore[union-attr]
            if shot.start_sec <= o <= shot.end_sec
        ]

        # Check if shot boundary aligns with a beat (within 0.1s tolerance)
        beat_aligned = any(
            abs(b - shot.start_sec) < 0.1 or abs(b - shot.end_sec) < 0.1
            for b in self._beats  # type: ignore[union-attr]
        )

        # Determine music cue
        music_cue = None
        if beats_in_shot:
            music_cue = "beat"

        return RhythmResult(
            bpm=round(self._bpm, 1),
            beat_aligned=beat_aligned,
            music_cue=music_cue,
            beats_in_shot=beats_in_shot,
            onsets_in_shot=onsets_in_shot,
        )

    def analyze_batch(self, shots: list[Shot], context: AnalysisContext) -> list[RhythmResult]:
        return [self.analyze_shot(shot, context) for shot in shots]

    def cleanup(self) -> None:
        self._audio = None
        self._beats = None
        self._onsets = None
        self._sr = 0
        self._bpm = 0.0
        logger.info("RhythmAnalyzer: audio data released")
