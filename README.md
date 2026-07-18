# Closed-Loop Music Therapy — Trained Inference Engine + Streamlit App

## What's in here

```
data_pipeline.py          # WESAD loader (real + synthetic fallback), windowing, feature extraction
train_model.py             # Trains RandomForest baseline + CNN-LSTM, saves to ./models
model_loader.py             # Unified inference wrapper used by the app (auto-picks CNN-LSTM or RF)
templates.py                 # HTML/CSS for the stress alert card, recovery card, track cards
app.py                        # Streamlit dashboard (st.fragment-based, non-blocking track switching)
build_notebook.py              # Generates notebooks/wesad_training_pipeline.ipynb
notebooks/wesad_training_pipeline.ipynb   # Deliverable notebook (same pipeline, walkthrough form)
requirements.txt
```

## 1. Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

On Apple Silicon, swap `tensorflow` for `tensorflow-macos` + `tensorflow-metal` in requirements.txt.

## 2. Get WESAD (optional but recommended for real paper results)

Download from: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
Unzip so you have:

```
data/WESAD/S2/S2.pkl
data/WESAD/S3/S3.pkl
...
```

**If you skip this step**, `train_model.py` automatically falls back to synthetic
physiological data (clearly labeled `"data_source": "synthetic"` in
`models/metrics.json`) so you can validate the whole pipeline immediately.
Do **not** report synthetic-data metrics in the paper — re-run with real WESAD first.

## 3. Train

```bash
python train_model.py                        # synthetic dev data, both models
python train_model.py --wesad data/WESAD      # real WESAD, both models
python train_model.py --skip-deep             # RF only (no TensorFlow needed)
```

Produces in `./models`:
- `rf_baseline.pkl`
- `cnn_lstm_stress.h5` + `cnn_lstm_norm_stats.pkl`
- `metrics.json` (accuracy/F1 for both)

## 4. Run the app

```bash
streamlit run app.py
```

The app auto-loads whichever model is available (CNN-LSTM preferred, RF fallback).

## How the track-switching bug was fixed

The old prototype used a blocking loop tied to the page's rerun cycle, so any
sidebar interaction (like picking a track) reset the stream. The fix:

- All stream state (`buffer_hr`, `buffer_eda`, `cursor`, `stress_label`,
  `recovery_mode`, ...) lives in `st.session_state`, which survives every rerun.
- The periodic advancement of the simulated telemetry + model inference is
  isolated in `@st.fragment(run_every=0.4)` — it reruns on its own 0.4s timer,
  independent of the rest of the page.
- The sidebar's track buttons only write to `st.session_state.current_track`
  (and, if relevant, flip `recovery_mode`). They never touch the fragment's
  loop, so switching between the 15 tracks never interrupts streaming.

## Human-in-the-loop recovery flow

1. Fragment computes rolling HRV/EDA window stats every tick → `model.predict()`.
2. If `label == 1` (stress), a custom alert card appears and low-tempo
   (<60 BPM) tracks in the sidebar get a "RECOMMENDED" badge.
3. If the user clicks one of those recommended tracks, `recovery_mode` turns
   on and subsequent simulated ticks are scaled down toward baseline
   (`recovery_scale` decays each cycle) — simulating biological down-regulation.
4. Once the model confirms the signal is back at baseline, a "Homeostasis
   Restored" card appears and `recovery_mode` clears automatically.

## Notes / limitations to disclose in the paper

- RMSSD is computed via `scipy.signal.find_peaks` R-peak detection, not a
  dedicated HRV library (e.g. NeuroKit2) — simpler but worth stating as a
  method choice/limitation.
- Train/test split is **subject-wise** (`GroupShuffleSplit` on subject IDs),
  not random-by-window, to avoid leakage — report those numbers, not a
  random split's (which will look artificially higher).
- The live app's telemetry is a **simulated** static-CSV stream (per the
  brief), not a live wearable connection.
