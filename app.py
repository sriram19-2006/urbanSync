from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.demo import LANES, DemoStep, build_demo_steps
from src.signal_controller import SignalController
from src.video_input import DetectionFrame, detect_vehicles_in_demo_frame, get_video_info
from src.visualizer import render_junction_frame


ROOT = Path(__file__).parent
VIDEO_DIR = ROOT / "videos"
MODEL_DIR = ROOT / "models"
YOLO_MODEL_NAME = "yolov8n.pt"
CONGESTION_COLORS = {
    "Low": "#34d399",
    "Moderate": "#fbbf24",
    "High": "#fb923c",
    "Severe": "#e11d48",
}
ACCENT_CYAN = "#22d3ee"
ACCENT_GREEN = "#34d399"


def main() -> None:
    st.set_page_config(page_title="AI Junction Optimizer", page_icon="traffic_light", layout="wide")
    _inject_styles()

    steps = build_demo_steps()
    _init_state(len(steps))

    with st.sidebar:
        st.title("AI Junction Optimizer")
        st.caption("Hackathon demo control")
        selected_video = _video_source_status()
        model = _load_yolo_model()
        st.divider()
        _step_controls(len(steps))

    detection = _detect_current_frame(selected_video, st.session_state.step_index, len(steps)) if selected_video else None
    step = _step_with_live_counts(steps[st.session_state.step_index], detection)
    chart_data = _chart_data(steps)

    st.markdown(
        """
        <div class="hero-block">
            <div class="hero-kicker">Smart-city traffic command center</div>
            <h1>AI Junction Optimizer</h1>
            <p>Adaptive signal control with YOLOv8 vehicle detection and emergency green corridor.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _summary_metrics(step, steps[-1])
    _main_dashboard(step, detection, model is not None)
    _charts(chart_data, steps[-1])
    _pitch_panel(step, steps[-1])


def _init_state(total_steps: int) -> None:
    if "step_index" not in st.session_state:
        st.session_state.step_index = 0
    st.session_state.step_index = min(max(st.session_state.step_index, 0), total_steps - 1)


def _video_source_status() -> Path | None:
    videos = sorted(
        path
        for path in VIDEO_DIR.glob("*")
        if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}
    )
    if not videos:
        st.info("No sample video file found yet. The dashboard is running on the built-in reliable demo feed.")
        return None

    videos = sorted(videos, key=lambda video: (video.name == "ai_junction_demo.mp4", video.name.lower()))
    selected_name = st.selectbox("Traffic video", [video.name for video in videos], index=0)
    selected_path = VIDEO_DIR / selected_name
    info = get_video_info(selected_path)
    if info is None:
        st.warning("This video could not be opened. The dashboard will keep using the reliable demo feed.")
        return None

    st.caption(
        f"Using {selected_name}: {info.width}x{info.height}, "
        f"{info.duration_seconds:.0f}s. The live preview samples this video during the demo."
    )
    return selected_path


@st.cache_resource(show_spinner="Loading YOLOv8 vehicle detector...")
def _load_yolo_model() -> object | None:
    # Streamlit keeps this model in memory, so it is not reloaded on every click.
    try:
        from ultralytics import YOLO
    except ImportError:
        return None

    model_path = MODEL_DIR / YOLO_MODEL_NAME
    source = str(model_path) if model_path.exists() else YOLO_MODEL_NAME
    try:
        return YOLO(source)
    except Exception:
        return None


@st.cache_data(show_spinner="Detecting vehicles with YOLOv8...")
def _detect_current_frame(video_path: Path | None, step_index: int, total_steps: int) -> DetectionFrame | None:
    if video_path is None:
        return None
    # Each slider step samples one frame. Streamlit caches the result for a smooth demo.
    return detect_vehicles_in_demo_frame(
        video_path=video_path,
        step_index=step_index,
        total_steps=total_steps,
        model=_load_yolo_model(),
    )


def _step_with_live_counts(base_step: DemoStep, detection: DetectionFrame | None) -> DemoStep:
    if detection is None or not detection.used_yolo:
        return base_step

    controller = SignalController(min_green=12, max_green=50)
    emergency_lane = base_step.emergency_lane if base_step.emergency_active else None
    decision = controller.choose_green_lane(detection.lane_counts, emergency_lane=emergency_lane)
    queue_total = sum(detection.lane_counts.values())

    return replace(
        base_step,
        lane_counts=detection.lane_counts,
        ai_green_lane=decision.green_lane,
        duration_seconds=decision.duration_seconds,
        reason=decision.reason,
        congestion_level=_congestion_level(queue_total),
        ai_queue_total=queue_total,
    )


def _step_controls(total_steps: int) -> None:
    selected_step = st.slider(
        "Inspection mode",
        min_value=1,
        max_value=total_steps,
        value=st.session_state.step_index + 1,
        format="Step %d",
    )
    st.session_state.step_index = selected_step - 1

    previous_col, next_col, reset_col = st.columns(3)
    with previous_col:
        if st.button("Prev", use_container_width=True):
            st.session_state.step_index = max(0, st.session_state.step_index - 1)
            st.rerun()
    with next_col:
        if st.button("Next", use_container_width=True):
            st.session_state.step_index = min(total_steps - 1, st.session_state.step_index + 1)
            st.rerun()
    with reset_col:
        if st.button("Restart", use_container_width=True):
            st.session_state.step_index = 0
            st.rerun()


def _summary_metrics(step: DemoStep, final_step: DemoStep) -> None:
    cols = st.columns(4)
    with cols[0]:
        _metric_card("Current green", step.ai_green_lane.title(), ACCENT_GREEN)
    with cols[1]:
        _metric_card("Green duration", f"{step.duration_seconds}s", ACCENT_CYAN)
    with cols[2]:
        _congestion_card(step.congestion_level)
    with cols[3]:
        _metric_card("Wait saved", str(final_step.saved_wait), ACCENT_CYAN)

    if step.emergency_active:
        _status_banner("Emergency corridor active: ambulance lane has priority.", emergency=True)
    else:
        _status_banner("Adaptive control active: signal is serving the highest-pressure lane.", emergency=False)


def _main_dashboard(step: DemoStep, detection: DetectionFrame | None, model_ready: bool) -> None:
    video_col, ai_col, right = st.columns([1.05, 1.35, 0.95])
    with video_col:
        st.markdown('<div class="section-heading">Traffic Video Input</div>', unsafe_allow_html=True)
        if detection is None:
            st.info("Add a traffic video to the videos folder to show real road footage here.")
        elif not model_ready:
            st.image(detection.frame, channels="RGB", use_container_width=True)
            st.warning("YOLOv8 is not available, so live vehicle boxes are paused.")
        else:
            st.image(detection.frame, channels="RGB", use_container_width=True)
            st.caption(f"Detected {len(detection.detections)} vehicles in this frame.")

    with ai_col:
        st.markdown('<div class="section-heading section-heading-primary">AI Junction Output</div>', unsafe_allow_html=True)
        st.image(render_junction_frame(step), channels="RGB", use_container_width=True)

    with right:
        st.markdown('<div class="section-heading">Lane Vehicle Counts</div>', unsafe_allow_html=True)
        max_count = max(max(step.lane_counts.values()), 1)
        for lane in LANES:
            count = step.lane_counts[lane]
            _lane_count_row(lane, count, count / max_count, lane == step.ai_green_lane)

        st.markdown('<div class="section-heading compact-heading">Signal Decision</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="signal-card">
                <div>AI selected <strong>{step.ai_green_lane.title()}</strong> for <strong>{step.duration_seconds} seconds</strong>.</div>
                <div>Reason: <strong>{step.reason.replace('_', ' ')}</strong>.</div>
                <div>Fixed timer would serve <strong>{step.fixed_green_lane.title()}</strong>.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _charts(chart_data: pd.DataFrame, final_step: DemoStep) -> None:
    st.markdown('<div class="section-heading chart-heading">Fixed Timer vs AI Timer</div>', unsafe_allow_html=True)
    comparison = pd.DataFrame(
        {
            "Mode": ["AI adaptive timer", "Fixed timer"],
            "Total wait": [final_step.ai_wait_total, final_step.fixed_wait_total],
            "Queue left": [final_step.ai_queue_total, final_step.fixed_queue_total],
        }
    )
    comparison_long = comparison.melt("Mode", var_name="Metric", value_name="Value")

    chart_col, detail_col = st.columns([1.2, 1])
    with chart_col:
        st.altair_chart(_bar_chart(comparison_long), use_container_width=True)
    with detail_col:
        st.altair_chart(_line_chart(chart_data, ["AI queue", "Fixed queue"], height=190), use_container_width=True)
        st.caption("Lower queue means vehicles are clearing faster.")

    st.altair_chart(_line_chart(chart_data, ["AI cumulative wait", "Fixed cumulative wait"], height=210), use_container_width=True)


def _metric_card(label: str, value: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="--accent:{accent};">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _lane_count_row(lane: str, count: int, ratio: float, is_green: bool) -> None:
    width = max(2, min(100, ratio * 100))
    active_class = " lane-active" if is_green else ""
    suffix = " green" if is_green else ""
    st.markdown(
        f"""
        <div class="lane-row{active_class}">
            <div class="lane-row-label">
                <span>{lane.title()} lane</span>
                <strong>{count} vehicles{suffix}</strong>
            </div>
            <div class="lane-track">
                <div class="lane-fill" style="width:{width:.1f}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _congestion_card(level: str) -> None:
    color = CONGESTION_COLORS.get(level, "#9ca3af")
    st.markdown(
        f"""
        <div class="congestion-card">
            <div class="congestion-label">Congestion</div>
            <div class="congestion-value">{level}</div>
            <div class="congestion-badge">
                <span class="congestion-dot" style="background:{color};"></span>
                <span>{level}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _congestion_level(total_queue: int) -> str:
    if total_queue >= 31:
        return "Severe"
    if total_queue >= 19:
        return "High"
    if total_queue >= 9:
        return "Moderate"
    return "Low"


def _status_banner(message: str, emergency: bool) -> None:
    class_name = "status-banner emergency-banner" if emergency else "status-banner normal-banner"
    st.markdown(f'<div class="{class_name}">{message}</div>', unsafe_allow_html=True)


def _pitch_panel(step: DemoStep, final_step: DemoStep) -> None:
    emergency_state = "Green corridor active" if step.emergency_active else "No emergency override"
    strategy = "Emergency priority" if step.emergency_active else "Serve highest-pressure lane"
    with st.expander("System Overview", expanded=False):
        st.markdown(
            f"""
            <div class="overview-list">
                <div><span>Current congestion status</span><strong>{step.congestion_level}</strong></div>
                <div><span>Active optimization reason</span><strong>{step.reason.replace('_', ' ')}</strong></div>
                <div><span>Estimated wait reduction</span><strong>{final_step.saved_wait} vehicle-steps</strong></div>
                <div><span>Emergency handling state</span><strong>{emergency_state}</strong></div>
                <div><span>AI traffic strategy</span><strong>{strategy}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _bar_chart(data: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Mode:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Value:Q", title=None),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(range=[ACCENT_GREEN, ACCENT_CYAN]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            xOffset="Metric:N",
            tooltip=["Mode:N", "Metric:N", "Value:Q"],
        )
        .properties(height=220)
        .configure_axis(labelColor="#b6c2d1", titleColor="#dbeafe", gridColor="#253041")
        .configure_legend(labelColor="#dbeafe")
        .configure_view(stroke="transparent")
        .configure(background="transparent")
    )


def _line_chart(data: pd.DataFrame, columns: list[str], height: int) -> alt.Chart:
    chart_data = data[["Step", *columns]].melt("Step", var_name="Metric", value_name="Value")
    return (
        alt.Chart(chart_data)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("Step:Q", title=None, axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("Value:Q", title=None),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(range=[ACCENT_GREEN, ACCENT_CYAN]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["Step:Q", "Metric:N", "Value:Q"],
        )
        .properties(height=height)
        .configure_axis(labelColor="#b6c2d1", titleColor="#dbeafe", gridColor="#253041")
        .configure_legend(labelColor="#dbeafe")
        .configure_view(stroke="transparent")
        .configure(background="transparent")
    )


@st.cache_data
def _chart_data(steps: list[DemoStep]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Step": step.step,
                "AI queue": step.ai_queue_total,
                "Fixed queue": step.fixed_queue_total,
                "AI cumulative wait": step.ai_wait_total,
                "Fixed cumulative wait": step.fixed_wait_total,
            }
            for step in steps
        ]
    )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #070b12;
            --panel: #0d1420;
            --panel-soft: #111a28;
            --panel-strong: #142033;
            --border: rgba(34, 211, 238, 0.18);
            --border-strong: rgba(52, 211, 153, 0.34);
            --text: #e5f0ff;
            --muted: #94a3b8;
            --cyan: #22d3ee;
            --green: #34d399;
        }
        html, body, [class*="css"] {
            font-family: "Inter", "Segoe UI", Arial, sans-serif;
        }
        .stApp {
            background:
                radial-gradient(circle at 18% 4%, rgba(34, 211, 238, 0.10), transparent 28%),
                radial-gradient(circle at 86% 16%, rgba(52, 211, 153, 0.08), transparent 30%),
                linear-gradient(135deg, #070b12 0%, #0a101a 46%, #081018 100%);
            color: var(--text);
        }
        .block-container {
            padding-top: 2.8rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        header[data-testid="stHeader"] {
            background: rgba(7, 11, 18, 0.78);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(34, 211, 238, 0.08);
        }
        h1, h2, h3, p, label, span, div {
            letter-spacing: 0;
        }
        .hero-block {
            margin-bottom: 1rem;
        }
        .hero-kicker {
            color: var(--green);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }
        .hero-block h1 {
            color: var(--text);
            font-size: 2.35rem;
            line-height: 1.05;
            margin: 0;
            text-shadow: 0 0 22px rgba(34, 211, 238, 0.12);
        }
        .hero-block p {
            color: var(--muted);
            font-size: 0.98rem;
            margin: 0.55rem 0 0;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #090e17 0%, #0b111c 100%);
            border-right: 1px solid rgba(34, 211, 238, 0.14);
            box-shadow: 8px 0 28px rgba(0,0,0,0.24);
            min-width: 270px !important;
            max-width: 270px !important;
        }
        [data-testid="stSidebarContent"] {
            padding: 1.35rem 1.1rem;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: var(--text) !important;
        }
        [data-testid="stSidebar"] h1 {
            font-size: 1.35rem;
            margin-bottom: 0.1rem;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: var(--muted) !important;
            font-size: 0.83rem;
        }
        [data-testid="stSidebar"] label p {
            color: var(--text) !important;
            font-size: 0.86rem;
            font-weight: 700;
        }
        [data-testid="stSidebar"] hr {
            border-color: rgba(148, 163, 184, 0.16);
            margin: 0.95rem 0;
        }
        [data-testid="stSelectbox"] > div {
            background: #0d1420;
            border: 1px solid rgba(34, 211, 238, 0.18);
            border-radius: 8px;
        }
        div[data-baseweb="select"] > div {
            background: #0d1420;
            border-color: rgba(34, 211, 238, 0.26);
            color: var(--text);
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(17, 26, 40, 0.96), rgba(10, 16, 26, 0.96));
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 0 20px rgba(34, 211, 238, 0.07);
        }
        div[data-testid="stMetric"] * {
            color: var(--text) !important;
        }
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, var(--green), var(--cyan));
        }
        .stProgress > div > div {
            background: #1b2637;
        }
        .metric-card,
        .congestion-card {
            position: relative;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(17, 26, 40, 0.98), rgba(9, 15, 25, 0.98));
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 16px;
            height: 112px;
            min-height: 112px;
            box-sizing: border-box;
            box-shadow: 0 0 20px rgba(34, 211, 238, 0.08), inset 0 1px 0 rgba(255,255,255,0.04);
        }
        .metric-card:before,
        .congestion-card:before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            height: 2px;
            width: 100%;
            background: linear-gradient(90deg, var(--accent, var(--cyan)), transparent);
        }
        .metric-label,
        .congestion-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 0.48rem;
            line-height: 1.25;
        }
        .metric-value,
        .congestion-value {
            color: var(--text);
            font-size: 1.55rem;
            line-height: 1.1;
            font-weight: 700;
            margin-bottom: 0.42rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .section-heading {
            color: var(--text);
            font-size: 1.12rem;
            line-height: 1.2;
            font-weight: 800;
            margin: 0.55rem 0 0.65rem;
            text-transform: uppercase;
        }
        .section-heading:after {
            content: "";
            display: block;
            width: 42px;
            height: 2px;
            margin-top: 7px;
            background: linear-gradient(90deg, var(--cyan), transparent);
        }
        .section-heading-primary {
            color: var(--cyan);
            text-shadow: 0 0 16px rgba(34, 211, 238, 0.22);
        }
        .compact-heading {
            margin-top: 1rem;
            font-size: 0.95rem;
        }
        .chart-heading {
            margin-top: 1.25rem;
        }
        [data-testid="stImage"] img {
            border-radius: 8px;
            border: 1px solid rgba(34, 211, 238, 0.20);
            box-shadow: 0 14px 38px rgba(0,0,0,0.34);
            background: #0b111c;
        }
        [data-testid="stColumn"]:nth-of-type(2) [data-testid="stImage"] img {
            border-color: rgba(52, 211, 153, 0.42);
            box-shadow: 0 0 0 1px rgba(52, 211, 153, 0.08), 0 0 30px rgba(52, 211, 153, 0.14), 0 18px 42px rgba(0,0,0,0.38);
        }
        .status-banner {
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin: 0.9rem 0 0.75rem;
            font-weight: 700;
            letter-spacing: 0;
        }
        .normal-banner {
            border: 1px solid rgba(52, 211, 153, 0.30);
            background: linear-gradient(90deg, rgba(20, 83, 45, 0.62), rgba(13, 20, 32, 0.72));
            color: #bbf7d0;
            box-shadow: 0 0 24px rgba(52, 211, 153, 0.10);
        }
        .emergency-banner {
            border: 1px solid rgba(248, 113, 113, 0.64);
            background: linear-gradient(90deg, rgba(127, 29, 29, 0.88), rgba(69, 10, 10, 0.78));
            color: #fee2e2;
            box-shadow: 0 0 26px rgba(239, 68, 68, 0.24), inset 3px 0 0 #ef4444;
        }
        .stButton > button {
            background: rgba(13, 20, 32, 0.72);
            color: #cbd5e1;
            border: 1px solid rgba(34, 211, 238, 0.18);
            border-radius: 7px;
            padding: 0.22rem 0.28rem;
            min-height: 1.78rem;
            font-size: 0.68rem;
            font-weight: 700;
            box-shadow: none;
            white-space: nowrap;
        }
        .stButton > button:hover {
            border-color: var(--cyan);
            color: #ffffff;
            background: rgba(19, 32, 51, 0.88);
        }
        [data-testid="stSidebar"] .stButton > button {
            min-height: 1.72rem;
            padding: 0.16rem 0.16rem;
            background: rgba(10, 16, 26, 0.55);
        }
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:nth-child(2) .stButton > button {
            background: linear-gradient(90deg, rgba(52, 211, 153, 0.28), rgba(34, 211, 238, 0.24));
            border-color: rgba(34, 211, 238, 0.46);
            color: #ecfeff;
            box-shadow: 0 0 14px rgba(34, 211, 238, 0.12);
        }
        .stSlider [data-baseweb="slider"] {
            margin-top: 0.04rem;
        }
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
            gap: 0.32rem;
            margin-top: 0.35rem;
            display: grid !important;
            grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
        }
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div {
            min-width: 0 !important;
            width: 100% !important;
        }
        .lane-row {
            margin-bottom: 0.72rem;
        }
        .lane-row-label {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 10px;
            color: var(--muted);
            font-size: 0.9rem;
            margin-bottom: 0.35rem;
        }
        .lane-row-label strong {
            color: var(--text);
            font-size: 0.88rem;
            font-weight: 700;
            white-space: nowrap;
        }
        .lane-track {
            height: 8px;
            background: #1b2637;
            border-radius: 999px;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.08);
        }
        .lane-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--green), var(--cyan));
            box-shadow: 0 0 12px rgba(34, 211, 238, 0.22);
        }
        .lane-active .lane-row-label strong {
            color: var(--green);
        }
        .signal-card {
            background: rgba(13, 20, 32, 0.72);
            border: 1px solid rgba(34, 211, 238, 0.14);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            color: var(--muted);
            line-height: 1.55;
            font-size: 0.9rem;
        }
        .signal-card strong {
            color: var(--text);
        }
        code {
            background: #080d15 !important;
            color: var(--green) !important;
            border: 1px solid rgba(52, 211, 153, 0.12);
        }
        .congestion-card {
            --accent: #22d3ee;
        }
        .congestion-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 0.82rem;
            font-weight: 700;
            color: var(--text);
            white-space: nowrap;
        }
        .congestion-dot {
            width: 12px;
            height: 12px;
            border-radius: 999px;
            box-shadow: inset 0 1px 2px rgba(255,255,255,0.55), 0 0 10px currentColor;
            display: inline-block;
        }
        .stMarkdown, .stCaptionContainer, p {
            color: var(--muted);
        }
        [data-testid="stExpander"] {
            background: rgba(13, 20, 32, 0.74);
            border: 1px solid rgba(34, 211, 238, 0.16);
            border-radius: 8px;
        }
        .overview-list {
            display: grid;
            gap: 0.55rem;
            padding: 0.25rem 0 0.1rem;
        }
        .overview-list div {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.10);
            padding-bottom: 0.42rem;
        }
        .overview-list span {
            color: var(--muted);
            font-size: 0.9rem;
        }
        .overview-list strong {
            color: var(--text);
            font-size: 0.9rem;
            text-align: right;
        }
        iframe {
            border-radius: 8px;
            background: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
