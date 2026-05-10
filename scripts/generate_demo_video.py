from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.visualizer import write_demo_video


if __name__ == "__main__":
    path = write_demo_video(ROOT / "videos" / "ai_junction_demo.mp4")
    print(f"Generated {path}")
