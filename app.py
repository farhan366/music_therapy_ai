"""
app.py
======
Human-in-the-Loop Closed-Loop Music Therapy Dashboard.

KEY FIX vs. the old prototype:
The old app broke/reset its telemetry stream whenever the user picked a new
track from the sidebar, because a plain `while True: ... st.rerun()` loop
tied the whole page's rerun cycle to the streaming loop -- any other widget
interaction (like clicking a track) forced a full script rerun that
re-initialized the loop's local variables.

The fix: all streaming state (buffer, cursor, stress flag, recovery flag)
lives in `st.session_state`, which persists across ANY rerun. The actual
periodic advancement of the stream is isolated inside an `st.fragment`
(`run_every=0.4`), so it reruns on its own timer independently of the rest
of the page. Selecting a track in the sidebar only touches
`st.session_state.current_track` -- it does not touch, reset, or block the
fragment's loop.

Run:
    streamlit run app.py
Requires a trained model in ./models (see train_model.py).
"""

import os
import time
import numpy as np
import pandas as pd
import streamlit as st

from model_loader import StressInferenceEngine
from templates import (
    STRESS_ALERT_CARD, RECOVERY_CARD, STATUS_BADGE_STRESS, STATUS_BADGE_BASELINE,
    track_card_html, metric_tile_html,
)

st.set_page_config(page_title="Closed-Loop Music Therapy", layout="wide")

# ---------------------------------------------------------------------------
# 1. TRACK LIBRARY (15 tracks; low-BPM ones are the "recommended" recovery set)
# ---------------------------------------------------------------------------

TRACK_LIBRARY = [
    {"name": "Deep Focus Drift", "bpm": 52, "genre": "Ambient"},
    {"name": "Clair de Lune (arr.)", "bpm": 54, "genre": "Classical"},
    {"name": "Slow Tide", "bpm": 48, "genre": "Ambient"},
    {"name": "Gymnopédie No.1 (arr.)", "bpm": 50, "genre": "Classical"},
    {"name": "Quiet Room", "bpm": 45, "genre": "Ambient"},
    {"name": "Soft Rain Piano", "bpm": 58, "genre": "Classical"},
    {"name": "Midnight Runner", "bpm": 128, "genre": "Electronic"},
    {"name": "Pulse Drive", "bpm": 140, "genre": "EDM"},
    {"name": "Neon Sprint", "bpm": 132, "genre": "Synthwave"},
    {"name": "City Lights", "bpm": 118, "genre": "Pop"},
    {"name": "Morning Coffee Jazz", "bpm": 96, "genre": "Jazz"},
    {"name": "Focus Flow", "bpm": 90, "genre": "Lo-fi"},
    {"name": "Study Beats", "bpm": 85, "genre": "Lo-fi"},
    {"name": "Warm Static", "bpm": 60, "genre": "Ambient"},
    {"name": "Golden Hour", "bpm": 70, "genre": "Indie"},
]
LOW_TEMPO_THRESHOLD = 60


# ---------------------------------------------------------------------------
# 2. SIMULATED TELEMETRY SOURCE (static CSV, streamed at 0.4s intervals)
# ---------------------------------------------------------------------------

@st.cache_data
def load_or_generate_telemetry_csv(path="data/sample_telemetry.csv", n_ticks=3000, seed=7):
    """Simulate a wearable's live feed as a static CSV (per the brief:
    'simulates live wearable sensor data streaming from a static CSV file').
    Instantaneous HR (bpm) and EDA (microsiemens) per tick, with slow
    baseline<->stress cycling so the demo naturally shows both states.
    """
    if os.path.exists(path):
        return pd.read_csv(path)

    rng = np.random.default_rng(seed)
    rows = []
    cycle_len = 150  # ticks per baseline/stress half-cycle (~60s at 0.4s/tick)
    for i in range(n_ticks):
        in_stress_phase = (i // cycle_len) % 2 == 1
        base_hr, stress_hr = 68, 100
        base_eda, stress_eda = 2.0, 7.0
        target_hr = stress_hr if in_stress_phase else base_hr
        target_eda = stress_eda if in_stress_phase else base_eda
        hr = target_hr + rng.normal(0, 3)
        eda = target_eda + rng.normal(0, 0.3)
        rows.append({"tick": i, "hr": max(40, hr), "eda": max(0.1, eda)})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# 3. SESSION STATE INITIALIZATION (persists across ALL reruns/interactions)
# ---------------------------------------------------------------------------

def init_session_state():
    defaults = {
        "cursor": 0,
        "buffer_hr": [],
        "buffer_eda": [],
        "window_ticks": 50,          # ~20s at 0.4s/tick
        "current_track": TRACK_LIBRARY[6]["name"],  # start on a normal track
        "stress_label": 0,
        "stress_proba": 0.0,
        "recovery_mode": False,
        "recovery_scale": 1.0,
        "streaming": True,
        "history": [],  # for the live chart
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if "engine" not in st.session_state:
        st.session_state.engine = StressInferenceEngine(prefer="cnn_lstm")


# ---------------------------------------------------------------------------
# 4. LIVE TELEMETRY + INFERENCE FRAGMENT
#    Isolated with st.fragment so its own rerun timer never gets disrupted
#    by sidebar interactions (track switching) elsewhere on the page.
# ---------------------------------------------------------------------------

@st.fragment(run_every=0.4)
def telemetry_fragment():
    telemetry_df = load_or_generate_telemetry_csv()
    n = len(telemetry_df)

    if st.session_state.streaming:
        row = telemetry_df.iloc[st.session_state.cursor % n]
        hr, eda = float(row["hr"]), float(row["eda"])

        # --- Biometric back-regulation ---
        # If the user has acted on the recommendation (switched to a
        # low-tempo track after a stress alert), progressively scale the
        # incoming signal down toward baseline to simulate the body
        # physically calming down over the next few cycles.
        if st.session_state.recovery_mode:
            st.session_state.recovery_scale = max(0.15, st.session_state.recovery_scale - 0.12)
            baseline_hr, baseline_eda = 68, 2.0
            hr = baseline_hr + (hr - baseline_hr) * st.session_state.recovery_scale
            eda = baseline_eda + (eda - baseline_eda) * st.session_state.recovery_scale

        st.session_state.buffer_hr.append(hr)
        st.session_state.buffer_eda.append(eda)
        st.session_state.buffer_hr = st.session_state.buffer_hr[-st.session_state.window_ticks:]
        st.session_state.buffer_eda = st.session_state.buffer_eda[-st.session_state.window_ticks:]
        st.session_state.cursor += 1

        st.session_state.history.append({"tick": st.session_state.cursor, "hr": hr, "eda": eda})
        st.session_state.history = st.session_state.history[-150:]

        # --- Live inference once buffer is full enough ---
        if len(st.session_state.buffer_hr) >= 10:
            hr_arr = np.array(st.session_state.buffer_hr)
            eda_arr = np.array(st.session_state.buffer_eda)
            rr_ms = 60000.0 / hr_arr
            rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))) if len(rr_ms) > 1 else 0.0

            engine = st.session_state.engine
            if engine.backend == "rf":
                label, proba = engine.predict_from_features(
                    hr_mean=float(hr_arr.mean()), hr_std=float(hr_arr.std()), rmssd=rmssd,
                    eda_mean=float(eda_arr.mean()), eda_max=float(eda_arr.max()), eda_std=float(eda_arr.std()),
                )
            else:
                # CNN-LSTM expects raw-ish waveform windows; here we
                # up-sample the buffered HR/EDA stats stream as a stand-in
                # signal so the same fragment works with either backend.
                label, proba = engine.predict_window(hr_arr, eda_arr)

            st.session_state.stress_label = label
            st.session_state.stress_proba = proba

            # Auto-verify homeostatic stabilization: once the model confirms
            # baseline again, exit recovery mode.
            if st.session_state.recovery_mode and label == 0 and st.session_state.recovery_scale <= 0.2:
                st.session_state.recovery_mode = False
                st.session_state.recovery_scale = 1.0
                st.session_state.just_recovered = True

    render_dashboard()


def render_dashboard():
    is_stress = st.session_state.stress_label == 1

    badge = STATUS_BADGE_STRESS if is_stress else STATUS_BADGE_BASELINE
    st.markdown(badge, unsafe_allow_html=True)

    if is_stress:
        st.markdown(STRESS_ALERT_CARD.format(confidence=st.session_state.stress_proba), unsafe_allow_html=True)
    elif st.session_state.get("just_recovered"):
        st.markdown(RECOVERY_CARD, unsafe_allow_html=True)
        st.session_state.just_recovered = False

    col1, col2, col3, col4 = st.columns(4)
    hr_now = st.session_state.buffer_hr[-1] if st.session_state.buffer_hr else 0
    eda_now = st.session_state.buffer_eda[-1] if st.session_state.buffer_eda else 0
    with col1:
        st.markdown(metric_tile_html("Heart Rate", f"{hr_now:.0f}", "bpm", "#ff6b6b"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_tile_html("EDA", f"{eda_now:.2f}", "µS", "#7597de"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_tile_html("Stress Probability", f"{st.session_state.stress_proba*100:.0f}", "%",
                                      "#ff6b6b" if is_stress else "#4caf82"), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_tile_html("Now Playing", st.session_state.current_track[:14], "", "#a880ff"),
                    unsafe_allow_html=True)

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history).set_index("tick")
        st.line_chart(hist_df, height=220)


# ---------------------------------------------------------------------------
# 5. SIDEBAR: track library (switching tracks does NOT touch the stream)
# ---------------------------------------------------------------------------

def render_sidebar():
    st.sidebar.title("🎵 Track Library")
    st.sidebar.caption("Selecting a track never interrupts the live telemetry stream.")

    is_stress = st.session_state.stress_label == 1
    recommended_names = {t["name"] for t in TRACK_LIBRARY if t["bpm"] < LOW_TEMPO_THRESHOLD}

    for track in TRACK_LIBRARY:
        is_recommended = is_stress and track["name"] in recommended_names
        st.sidebar.markdown(
            track_card_html(track["name"], track["bpm"], track["genre"], is_recommended),
            unsafe_allow_html=True,
        )
        if st.sidebar.button(f"Play '{track['name']}'", key=f"play_{track['name']}", use_container_width=True):
            st.session_state.current_track = track["name"]
            # Human-in-the-loop trigger: if the user reacts to a stress
            # alert by picking a recommended low-tempo track, kick off the
            # biometric back-regulation simulation.
            if is_stress and track["name"] in recommended_names:
                st.session_state.recovery_mode = True
                st.session_state.recovery_scale = 1.0

    st.sidebar.divider()
    st.session_state.streaming = st.sidebar.toggle("Live telemetry streaming", value=st.session_state.streaming)
    if st.sidebar.button("Reset session"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    init_session_state()
    st.title("🧠 Closed-Loop Music Therapy — Live Dashboard")
    st.caption(
        f"Model backend: **{st.session_state.engine.backend.upper()}** · "
        "Streaming simulated wearable telemetry every 0.4s."
    )
    render_sidebar()
    telemetry_fragment()


if __name__ == "__main__":
    main()
