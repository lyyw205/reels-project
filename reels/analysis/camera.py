"""Camera motion analyzer using optical flow."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from reels.analysis.base import AnalysisContext
from reels.models.analysis import CameraResult
from reels.models.shot import Shot

logger = logging.getLogger(__name__)


class CameraAnalyzer:
    """Classify camera motion type using Farneback optical flow."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("analysis", {}).get("camera", {})
        self.sample_interval: int = cfg.get("sample_interval_frames", 2)
        self.motion_threshold: float = cfg.get("motion_threshold", 2.0)

    @property
    def name(self) -> str:
        return "camera"

    def analyze_shot(self, shot: Shot, context: AnalysisContext) -> CameraResult:
        """Analyze camera motion for a single shot."""
        cap = cv2.VideoCapture(str(context.video_path))
        if not cap.isOpened():
            return CameraResult()

        cap.set(cv2.CAP_PROP_POS_FRAMES, shot.start_frame)
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return CameraResult()

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        flow_vectors: list[np.ndarray] = []
        magnitudes: list[float] = []
        radial_divergences: list[float] = []

        frame_idx = shot.start_frame + self.sample_interval
        while frame_idx < shot.end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, curr_frame = cap.read()
            if not ret:
                break

            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )

            avg_flow = flow.mean(axis=(0, 1))
            mag = float(np.sqrt(avg_flow[0] ** 2 + avg_flow[1] ** 2))

            flow_vectors.append(avg_flow)
            magnitudes.append(mag)
            radial_divergences.append(self._compute_radial_divergence(flow))

            prev_gray = curr_gray
            frame_idx += self.sample_interval

        cap.release()

        if not flow_vectors:
            return CameraResult()

        avg_magnitude = float(np.mean(magnitudes))
        dominant_flow = np.mean(flow_vectors, axis=0)
        shake_score = self._compute_shake_score(flow_vectors)
        avg_divergence = float(np.mean(radial_divergences))
        camera_type = self._classify_motion(
            dominant_flow, avg_magnitude, shake_score, avg_divergence,
        )

        return CameraResult(
            camera_type=camera_type,
            shake_score=shake_score,
            avg_magnitude=round(avg_magnitude, 3),
            dominant_direction=[round(float(dominant_flow[0]), 3), round(float(dominant_flow[1]), 3)],
        )

    def analyze_batch(self, shots: list[Shot], context: AnalysisContext) -> list[CameraResult]:
        return [self.analyze_shot(shot, context) for shot in shots]

    def cleanup(self) -> None:
        pass  # No heavy models to unload

    def _compute_shake_score(self, flow_vectors: list[np.ndarray]) -> float:
        """Compute shake score based on flow vector variance. Returns 0-1."""
        if len(flow_vectors) < 2:
            return 0.0
        arr = np.array(flow_vectors)
        variance = float(np.std(arr, axis=0).mean())
        # Normalize to 0-1 range (empirical: variance > 5 = very shaky)
        return min(1.0, variance / 5.0)

    @staticmethod
    def _compute_radial_divergence(flow: np.ndarray) -> float:
        """Compute radial divergence of flow field relative to frame center.

        Positive divergence = vectors point outward from center (push_in / zoom in).
        Negative divergence = vectors point inward toward center (pull_out / zoom out).
        """
        h, w = flow.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        # Build grid of pixel coordinates relative to center
        ys, xs = np.mgrid[0:h, 0:w]
        rx = (xs - cx).astype(np.float32)
        ry = (ys - cy).astype(np.float32)

        # Radial distance from center (avoid division by zero)
        r = np.sqrt(rx ** 2 + ry ** 2)
        r[r < 1e-6] = 1e-6

        # Unit radial direction
        rx_hat = rx / r
        ry_hat = ry / r

        # Dot product of flow with radial direction: positive = outward
        radial_component = flow[..., 0] * rx_hat + flow[..., 1] * ry_hat

        return float(np.mean(radial_component))

    def _classify_motion(
        self,
        dominant_flow: np.ndarray,
        avg_magnitude: float,
        shake_score: float,
        avg_divergence: float = 0.0,
    ) -> str:
        """Classify camera motion type from optical flow statistics."""
        dx, dy = float(dominant_flow[0]), float(dominant_flow[1])

        if avg_magnitude < self.motion_threshold:
            return "static"

        if shake_score > 0.5:
            return "handheld"

        if shake_score < 0.15 and avg_magnitude > self.motion_threshold:
            # Smooth motion = gimbal
            abs_dx, abs_dy = abs(dx), abs(dy)

            if abs_dx > abs_dy * 1.5:
                return "pan_right" if dx > 0 else "pan_left"
            if abs_dy > abs_dx * 1.5:
                return "tilt_down" if dy > 0 else "tilt_up"

            # Radial divergence check for push-in / pull-out
            if abs(avg_divergence) > 0.5:
                return "push_in" if avg_divergence > 0 else "pull_out"

            return "gimbal_smooth"

        # Moderate shake + direction
        abs_dx, abs_dy = abs(dx), abs(dy)
        if abs_dx > abs_dy * 1.5:
            return "pan_right" if dx > 0 else "pan_left"
        if abs_dy > abs_dx * 1.5:
            return "tilt_down" if dy > 0 else "tilt_up"

        return "handheld"
