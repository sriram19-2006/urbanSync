# Smart Traffic Signal Simulator

A small Python project for experimenting with traffic signal timing from lane counts, simulated vehicle flow, and emergency vehicle priority.

## Project Layout

```text
README.md
TODO.md
app.py
requirements.txt
src/
  detector.py
  lane_counter.py
  signal_controller.py
  emergency.py
  simulator.py
  utils.py
sample_videos/
  .gitkeep
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run The Simulator

```bash
python app.py
```

The app runs a short command-line simulation and prints the selected signal phase at each step.

## Optional Video Input

Put traffic videos in `sample_videos/`. The current detector provides a lightweight OpenCV frame reader and placeholder vehicle-count logic that can be extended with a real object detection model.
