from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DetectionResult:
    """Vehicle count detected in one video frame."""

    vehicle_count: int
    frame_index: int


class VehicleDetector:
    """Lightweight detector scaffold for video-based traffic counting."""

    def count_from_frame(self, frame: np.ndarray, frame_index: int = 0) -> DetectionResult:
        if frame.size == 0:
            return DetectionResult(vehicle_count=0, frame_index=frame_index)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = [contour for contour in contours if cv2.contourArea(contour) > 600]

        return DetectionResult(vehicle_count=len(large_contours), frame_index=frame_index)

    def count_from_video(self, video_path: str | Path, max_frames: int = 120) -> list[DetectionResult]:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path}")

        results: list[DetectionResult] = []
        frame_index = 0

        try:
            while frame_index < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                results.append(self.count_from_frame(frame, frame_index))
                frame_index += 1
        finally:
            capture.release()

        return results
