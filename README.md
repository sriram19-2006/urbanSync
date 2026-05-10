# AI Junction Optimizer

A hackathon-ready dashboard for demonstrating AI-based traffic signal optimization at a road junction.

Live Vercel demo:

https://ai-junction-optimizer-demo.vercel.app

The demo shows:

- lane-wise vehicle counts
- YOLOv8 vehicle boxes on the selected traffic video
- dynamic green-signal selection
- signal duration based on congestion
- emergency green corridor behavior for an ambulance
- fixed-timer vs AI-timer comparison charts

## Setup

Use Python 3.11 on this machine because the plain `python` command points to a broken install.

```powershell
py -3.11 -m pip install -r requirements.txt
```

## Generate The Sample Video

```powershell
py -3.11 scripts\generate_demo_video.py
```

This creates:

```text
videos/ai_junction_demo.mp4
```

## Run The Dashboard

```powershell
py -3.11 -m streamlit run app.py
```

Open the local URL Streamlit prints in the terminal. Use the sidebar controls to step through the demo.

Put your own traffic videos inside the `videos/` folder. The dashboard will show frames from the selected video alongside the AI signal decision.

The project uses the small YOLOv8 model in `models/yolov8n.pt` to detect cars, buses, trucks, motorcycles, and bicycles.

## Test

```powershell
py -3.11 -m unittest discover -s tests
```

## Demo Pitch

Fixed traffic signals treat empty and crowded lanes the same. This system watches traffic pressure, adapts green time, and immediately gives ambulances a green corridor. The dashboard makes the benefit visible by comparing the AI timer against a normal fixed timer.
