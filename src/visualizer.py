from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.demo import DemoStep, LANES, build_demo_steps


LANE_LABELS = {
    "north": "NORTH",
    "east": "EAST",
    "south": "SOUTH",
    "west": "WEST",
}


def render_junction_frame(step: DemoStep, width: int = 960, height: int = 540) -> np.ndarray:
    frame = np.full((height, width, 3), (8, 13, 22), dtype=np.uint8)
    center_x, center_y = width // 2, height // 2
    road_color = (34, 45, 61)
    lane_line = (120, 224, 255)
    green = (52, 211, 153)
    red = (210, 71, 78)
    amber = (238, 169, 65)
    text = (226, 240, 255)

    cv2.rectangle(frame, (center_x - 120, 0), (center_x + 120, height), road_color, -1)
    cv2.rectangle(frame, (0, center_y - 90), (width, center_y + 90), road_color, -1)
    cv2.line(frame, (center_x, 0), (center_x, height), lane_line, 2)
    cv2.line(frame, (0, center_y), (width, center_y), lane_line, 2)
    cv2.rectangle(frame, (center_x - 120, center_y - 90), (center_x + 120, center_y + 90), (45, 58, 78), -1)

    _draw_title(frame, step)
    _draw_signal(frame, (center_x - 148, center_y - 118), step.ai_green_lane == "north", "N")
    _draw_signal(frame, (center_x + 116, center_y - 118), step.ai_green_lane == "east", "E")
    _draw_signal(frame, (center_x + 116, center_y + 84), step.ai_green_lane == "south", "S")
    _draw_signal(frame, (center_x - 148, center_y + 84), step.ai_green_lane == "west", "W")

    for lane in LANES:
        _draw_queue(frame, lane, step.lane_counts[lane], step.ai_green_lane == lane)

    if step.emergency_active and step.emergency_lane:
        _draw_ambulance(frame, step.emergency_lane)
        banner_left = 24
        banner_top = height - 118
        banner_right = width - 24
        banner_bottom = height - 58
        cv2.rectangle(frame, (banner_left, banner_top), (banner_right, banner_bottom), (42, 12, 18), -1)
        cv2.rectangle(frame, (banner_left, banner_top), (banner_right, banner_bottom), red, 2)
        cv2.putText(
            frame,
            "EMERGENCY GREEN CORRIDOR ACTIVE",
            (banner_left + 24, banner_top + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (255, 226, 226),
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(frame, (width - 310, 26), (width - 24, 126), (13, 20, 32), -1)
    cv2.rectangle(frame, (width - 310, 26), (width - 24, 126), (34, 211, 238), 1)
    cv2.putText(frame, "GREEN SIGNAL", (width - 288, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        LANE_LABELS[step.ai_green_lane],
        (width - 288, 101),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        green,
        3,
        cv2.LINE_AA,
    )

    if step.reason == "emergency vehicle priority":
        cv2.circle(frame, (width - 58, 56), 13, red, -1)
    else:
        cv2.circle(frame, (width - 58, 56), 13, amber, -1)

    return frame


def write_demo_video(path: str | Path, fps: int = 2) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [render_junction_frame(step) for step in build_demo_steps()]
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return output_path


def _draw_title(frame: np.ndarray, step: DemoStep) -> None:
    cv2.putText(frame, f"AI Junction Optimizer - Step {step.step:02d}", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (226, 240, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Detected queue: {sum(step.lane_counts.values())} vehicles", (24, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (148, 163, 184), 2, cv2.LINE_AA)


def _draw_signal(frame: np.ndarray, origin: tuple[int, int], active: bool, label: str) -> None:
    x, y = origin
    cv2.rectangle(frame, (x, y), (x + 32, y + 64), (31, 35, 42), -1)
    cv2.circle(frame, (x + 16, y + 18), 9, (60, 42, 48), -1)
    cv2.circle(frame, (x + 16, y + 46), 9, (33, 74, 52) if active else (49, 65, 58), -1)
    if active:
        cv2.circle(frame, (x + 16, y + 46), 9, (36, 204, 115), -1)
    else:
        cv2.circle(frame, (x + 16, y + 18), 9, (222, 71, 83), -1)
    cv2.putText(frame, label, (x + 38, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (226, 240, 255), 2, cv2.LINE_AA)


def _draw_queue(frame: np.ndarray, lane: str, count: int, moving: bool) -> None:
    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
    car_color = (52, 211, 153) if moving else (74, 94, 122)
    max_cars = min(count, 12)

    for i in range(max_cars):
        if lane == "north":
            x = center_x + 25
            y = center_y - 130 - i * 30
            rect = ((x, y), (x + 34, y + 20))
        elif lane == "south":
            x = center_x - 58
            y = center_y + 110 + i * 30
            rect = ((x, y), (x + 34, y + 20))
        elif lane == "east":
            x = center_x + 150 + i * 40
            y = center_y + 24
            rect = ((x, y), (x + 30, y + 20))
        else:
            x = center_x - 180 - i * 40
            y = center_y - 48
            rect = ((x, y), (x + 30, y + 20))

        cv2.rectangle(frame, rect[0], rect[1], car_color, -1)
        cv2.rectangle(frame, rect[0], rect[1], (34, 211, 238), 1)

    label_pos = {
        "north": (center_x + 70, 126),
        "east": (center_x + 235, center_y + 74),
        "south": (center_x - 104, height_minus(frame, 90)),
        "west": (60, center_y - 62),
    }[lane]
    cv2.putText(frame, f"{LANE_LABELS[lane]} {count}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (226, 240, 255), 2, cv2.LINE_AA)


def _draw_ambulance(frame: np.ndarray, lane: str) -> None:
    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
    if lane == "east":
        x, y = center_x + 98, center_y + 24
        cv2.rectangle(frame, (x, y), (x + 44, y + 24), (226, 240, 255), -1)
        cv2.rectangle(frame, (x, y), (x + 44, y + 24), (210, 71, 78), 2)
        cv2.line(frame, (x + 22, y + 5), (x + 22, y + 19), (210, 71, 78), 2)
        cv2.line(frame, (x + 15, y + 12), (x + 29, y + 12), (210, 71, 78), 2)


def height_minus(frame: np.ndarray, value: int) -> int:
    return frame.shape[0] - value
