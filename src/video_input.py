from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.demo import LANES


VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle"}
YOLO_CONFIDENCE = 0.08
YOLO_IMAGE_SIZE = 960
MIN_BOX_AREA = 18
LANE_COLORS = {
    "north": (45, 180, 255),
    "east": (46, 204, 113),
    "south": (255, 159, 67),
    "west": (155, 89, 182),
}


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    frame_count: int
    fps: float
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


@dataclass(frozen=True)
class VehicleDetection:
    label: str
    confidence: float
    lane: str
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectionFrame:
    frame: np.ndarray
    lane_counts: dict[str, int]
    detections: list[VehicleDetection]
    frame_index: int
    used_yolo: bool


def get_video_info(video_path: str | Path) -> VideoInfo | None:
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        return VideoInfo(
            path=path,
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


def read_demo_frame(video_path: str | Path, step_index: int, total_steps: int) -> np.ndarray | None:
    info = get_video_info(video_path)
    if info is None or info.frame_count <= 0:
        return None

    position = step_index / max(1, total_steps - 1)
    frame_index = min(info.frame_count - 1, int(position * (info.frame_count - 1)))

    capture = cv2.VideoCapture(str(info.path))
    try:
        if not capture.isOpened():
            return None
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            return None
    finally:
        capture.release()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return _add_label(frame, f"Traffic video input - frame {frame_index + 1:,}")


def detect_vehicles_in_demo_frame(
    video_path: str | Path,
    step_index: int,
    total_steps: int,
    model: object | None,
    confidence: float = YOLO_CONFIDENCE,
) -> DetectionFrame | None:
    frame_data = read_raw_demo_frame(video_path, step_index, total_steps)
    if frame_data is None:
        return None

    frame, frame_index = frame_data
    lane_counts = {lane: 0 for lane in LANES}
    detections: list[VehicleDetection] = []

    # If the model is unavailable, still show the raw video so the demo does not break.
    if model is None:
        labelled = _add_label(frame, f"Traffic video input - frame {frame_index + 1:,} | YOLO model unavailable")
        return DetectionFrame(labelled, lane_counts, detections, frame_index, used_yolo=False)

    # YOLO expects OpenCV-style BGR images. The dashboard uses RGB for display,
    # so convert before prediction to avoid confusing the model at night.
    yolo_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # The top-view night video has many tiny distant vehicles. A larger image
    # size catches more of them, while Streamlit caching keeps repeated clicks fast.
    results = model.predict(yolo_frame, imgsz=YOLO_IMAGE_SIZE, conf=confidence, iou=0.45, verbose=False)
    names = getattr(model, "names", {})

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = str(names.get(class_id, class_id))
            if label not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            if _box_area((x1, y1, x2, y2)) < MIN_BOX_AREA:
                continue
            confidence_score = float(box.conf[0])
            lane = _lane_for_box((x1, y1, x2, y2), frame.shape[1], frame.shape[0])
            lane_counts[lane] += 1
            detections.append(VehicleDetection(label, confidence_score, lane, (x1, y1, x2, y2)))

    annotated = _draw_detections(frame, detections)
    annotated = _draw_lane_guides(annotated)
    annotated = _add_label(annotated, f"YOLOv8 vehicle detection - frame {frame_index + 1:,}")
    return DetectionFrame(annotated, lane_counts, detections, frame_index, used_yolo=True)


def read_raw_demo_frame(video_path: str | Path, step_index: int, total_steps: int) -> tuple[np.ndarray, int] | None:
    info = get_video_info(video_path)
    if info is None or info.frame_count <= 0:
        return None

    position = step_index / max(1, total_steps - 1)
    frame_index = min(info.frame_count - 1, int(position * (info.frame_count - 1)))

    capture = cv2.VideoCapture(str(info.path))
    try:
        if not capture.isOpened():
            return None
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            return None
    finally:
        capture.release()

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame_index


def _add_label(frame: np.ndarray, label: str) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 44), (18, 22, 30), -1)
    cv2.putText(output, label, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def _lane_for_box(box: tuple[int, int, int, int], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    middle_x = width / 2
    middle_y = height / 2

    # Top-view junction videos usually map cleanly into four approach areas.
    # Whichever direction is farther from the center becomes the lane name.
    if abs(center_x - middle_x) > abs(center_y - middle_y):
        return "east" if center_x > middle_x else "west"
    return "south" if center_y > middle_y else "north"


def _box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _draw_detections(frame: np.ndarray, detections: list[VehicleDetection]) -> np.ndarray:
    output = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        color = LANE_COLORS[detection.lane]
        text = f"{detection.label} {detection.confidence:.2f} | {detection.lane}"
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(output, (x1, max(0, y1 - 24)), (min(output.shape[1], x1 + 178), y1), color, -1)
        cv2.putText(output, text, (x1 + 4, max(16, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def _draw_lane_guides(frame: np.ndarray) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    center_x = width // 2
    center_y = height // 2
    guide_color = (255, 255, 255)
    cv2.line(output, (center_x, 44), (center_x, height), guide_color, 1)
    cv2.line(output, (0, center_y), (width, center_y), guide_color, 1)
    cv2.putText(output, "N", (center_x + 8, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.58, guide_color, 2, cv2.LINE_AA)
    cv2.putText(output, "S", (center_x + 8, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.58, guide_color, 2, cv2.LINE_AA)
    cv2.putText(output, "W", (12, center_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, guide_color, 2, cv2.LINE_AA)
    cv2.putText(output, "E", (width - 28, center_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, guide_color, 2, cv2.LINE_AA)
    return output
