"""
data_pipeline.py
=================
WESAD ingestion, windowing, and feature engineering for the closed-loop
music-therapy stress classifier.

Two data sources are supported:

1. REAL WESAD DATASET (recommended for the paper)
   Download from: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
   Unzip so you have: data/WESAD/S2/S2.pkl, data/WESAD/S3/S3.pkl, ...
   Each subject .pkl is a dict with keys 'signal', 'label', 'subject'.
   We use the chest device (RespiBAN, 700 Hz) for ECG-derived HRV and the
   wrist device (Empatica E4) for EDA (4 Hz), matching what a real wearable
   pipeline would look like downstream.

2. SYNTHETIC FALLBACK
   If data/WESAD is not present, `load_wesad_or_synthetic()` generates
   physiologically plausible EDA/HRV traces labeled 0 (baseline) / 1 (stress)
   so the full pipeline (windowing -> features -> model) is runnable and
   testable immediately, without waiting on the ~9 GB download.

   IMPORTANT FOR THE PAPER: the synthetic data is only a development aid.
   For actual results/metrics you report, train on the real WESAD dataset.
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, resample

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# WESAD label codes (per dataset documentation)
# 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation ...
WESAD_BASELINE_LABELS = {1, 3, 4}   # collapse to class 0 (non-stress)
WESAD_STRESS_LABEL = 2              # class 1 (acute stress)

CHEST_FS = 700   # Hz, RespiBAN chest device
WRIST_EDA_FS = 4  # Hz, Empatica E4 wrist EDA


# ---------------------------------------------------------------------------
# 1. RAW SIGNAL LOADING
# ---------------------------------------------------------------------------

def load_subject_wesad(subject_path):
    """Load one WESAD subject .pkl and return chest ECG, wrist EDA, and labels.

    Parameters
    ----------
    subject_path : str
        Path to e.g. data/WESAD/S2/S2.pkl

    Returns
    -------
    ecg : np.ndarray, chest ECG @ 700 Hz
    eda : np.ndarray, wrist EDA @ 4 Hz (resampled to 700 Hz for alignment)
    labels : np.ndarray, per-sample label @ 700 Hz (chest sampling rate)
    """
    with open(subject_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    ecg = np.asarray(data["signal"]["chest"]["ECG"]).flatten()
    eda_wrist = np.asarray(data["signal"]["wrist"]["EDA"]).flatten()
    labels = np.asarray(data["label"]).flatten()

    # Resample wrist EDA (4 Hz) up to chest rate (700 Hz) so every stream
    # shares one common timeline before windowing.
    eda_resampled = resample(eda_wrist, len(ecg))

    return ecg, eda_resampled, labels


def load_wesad_dataset(wesad_root="data/WESAD", subjects=None):
    """Load and concatenate all requested WESAD subjects.

    Returns a dict: {'ecg': ..., 'eda': ..., 'labels': ..., 'subject_id': ...}
    subject_id is kept so you can do subject-wise (not just random) train/test
    splits, which is best practice for WESAD and worth mentioning in the paper
    to avoid inflated accuracy from subject leakage.
    """
    if subjects is None:
        subjects = [f"S{i}" for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]]

    all_ecg, all_eda, all_labels, all_subj = [], [], [], []
    for subj in subjects:
        subj_file = os.path.join(wesad_root, subj, f"{subj}.pkl")
        if not os.path.exists(subj_file):
            continue
        ecg, eda, labels = load_subject_wesad(subj_file)
        all_ecg.append(ecg)
        all_eda.append(eda)
        all_labels.append(labels)
        all_subj.append(np.full(len(ecg), subj))

    if not all_ecg:
        raise FileNotFoundError(
            f"No WESAD subject files found under '{wesad_root}'. "
            "Download the dataset and place per-subject .pkl files there, "
            "or use load_wesad_or_synthetic() for a synthetic dev/test run."
        )

    return {
        "ecg": np.concatenate(all_ecg),
        "eda": np.concatenate(all_eda),
        "labels": np.concatenate(all_labels),
        "subject_id": np.concatenate(all_subj),
    }


# ---------------------------------------------------------------------------
# 2. SYNTHETIC FALLBACK (for dev/testing without the 9GB download)
# ---------------------------------------------------------------------------

def generate_synthetic_session(duration_sec=600, fs=700, n_subjects=6, seed=RANDOM_SEED):
    """Generate synthetic multi-subject ECG-like + EDA-like signals with
    alternating baseline/stress segments, mimicking WESAD's protocol
    structure (baseline -> stress -> recovery, repeated).
    """
    rng = np.random.default_rng(seed)
    all_ecg, all_eda, all_labels, all_subj = [], [], [], []

    t = np.arange(0, duration_sec, 1 / fs)
    n_samples = len(t)

    for subj_idx in range(n_subjects):
        # Randomize per-subject baseline HR/EDA so the model must generalize.
        base_hr = rng.uniform(60, 75)       # bpm at rest
        stress_hr = base_hr + rng.uniform(20, 35)
        base_eda_level = rng.uniform(1.0, 3.0)   # microsiemens
        stress_eda_level = base_eda_level + rng.uniform(3.0, 6.0)

        # Build a label timeline: alternating 60s baseline / 60s stress blocks.
        # Use WESAD's own label codes (1=baseline, 2=stress) so this plugs
        # into the same windowing/discard logic as the real dataset.
        block_len = fs * 60
        labels = np.full(n_samples, 1, dtype=int)  # start baseline (code 1)
        toggle = 0
        for start in range(0, n_samples, block_len):
            end = min(start + block_len, n_samples)
            labels[start:end] = WESAD_STRESS_LABEL if toggle == 1 else 1
            toggle = 1 - toggle

        # ECG-like synthetic: instantaneous HR modulated by label, then
        # turned into a peaky waveform so our RMSSD/peak-detection code has
        # something realistic to chew on.
        instant_hr = np.where(labels == WESAD_STRESS_LABEL, stress_hr, base_hr).astype(float)
        instant_hr += rng.normal(0, 2.0, size=n_samples)  # natural variability
        # Simulate beat times via integrating instantaneous frequency
        beat_phase = np.cumsum(instant_hr / 60.0) / fs
        ecg = np.sin(2 * np.pi * beat_phase) ** 19  # peaky pseudo-QRS shape
        ecg += rng.normal(0, 0.02, size=n_samples)

        # EDA-like synthetic: slow-varying tonic level + label-driven shifts
        eda = np.where(labels == WESAD_STRESS_LABEL, stress_eda_level, base_eda_level).astype(float)
        # smooth transitions (EDA responds over seconds, not instantly)
        smooth_win = fs * 3
        eda = pd.Series(eda).rolling(smooth_win, min_periods=1, center=True).mean().to_numpy().copy()
        eda += rng.normal(0, 0.05, size=n_samples)

        all_ecg.append(ecg)
        all_eda.append(eda)
        all_labels.append(labels)
        all_subj.append(np.full(n_samples, f"SYN{subj_idx}"))

    return {
        "ecg": np.concatenate(all_ecg),
        "eda": np.concatenate(all_eda),
        "labels": np.concatenate(all_labels),
        "subject_id": np.concatenate(all_subj),
    }


def load_wesad_or_synthetic(wesad_root="data/WESAD"):
    """Try to load real WESAD; fall back to synthetic data with a printed
    warning so it's always obvious which source produced a given result.
    """
    try:
        data = load_wesad_dataset(wesad_root)
        print(f"[data_pipeline] Loaded REAL WESAD data from '{wesad_root}'.")
        return data, "real"
    except FileNotFoundError as e:
        print(f"[data_pipeline] {e}")
        print("[data_pipeline] Falling back to SYNTHETIC data for pipeline "
              "development/testing. Re-run with real WESAD before reporting "
              "results in the paper.")
        return generate_synthetic_session(), "synthetic"


# ---------------------------------------------------------------------------
# 3. WINDOWING + FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def _rr_intervals_from_ecg(ecg_window, fs):
    """Detect R-peaks in an ECG-like window and return RR intervals (ms)."""
    if len(ecg_window) < fs // 2:
        return np.array([])
    min_distance = int(fs * 0.4)  # refractory ~ <150bpm cap
    peaks, _ = find_peaks(ecg_window, distance=min_distance, prominence=0.3)
    if len(peaks) < 2:
        return np.array([])
    rr = np.diff(peaks) / fs * 1000.0  # ms
    return rr


def _hrv_features(ecg_window, fs):
    rr = _rr_intervals_from_ecg(ecg_window, fs)
    if len(rr) < 2:
        return {"hr_mean": np.nan, "hr_std": np.nan, "rmssd": np.nan}
    hr = 60000.0 / rr
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
    return {"hr_mean": float(np.mean(hr)), "hr_std": float(np.std(hr)), "rmssd": float(rmssd)}


def _eda_features(eda_window):
    return {
        "eda_mean": float(np.mean(eda_window)),
        "eda_max": float(np.max(eda_window)),
        "eda_std": float(np.std(eda_window)),
    }


def make_windows(ecg, eda, labels, fs=CHEST_FS, window_sec=20, overlap=0.5,
                  subject_id=None):
    """Slide a window_sec window (default 20s, 50% overlap) across the
    aligned ECG+EDA streams, extract statistical features per window, and
    assign the majority label within the window.

    Returns
    -------
    X : np.ndarray, shape (n_windows, n_features)   -- for the RF baseline
    X_seq : np.ndarray, shape (n_windows, window_len, 2) -- raw [ECG, EDA]
            sequences for the CNN-LSTM (spatial+temporal model)
    y : np.ndarray, shape (n_windows,)
    feature_names : list[str]
    groups : np.ndarray, subject id per window (for subject-wise CV/splits)
    """
    win_len = int(window_sec * fs)
    step = int(win_len * (1 - overlap))
    n_samples = len(ecg)

    feats, seqs, ys, groups = [], [], [], []
    feature_names = ["hr_mean", "hr_std", "rmssd", "eda_mean", "eda_max", "eda_std"]

    for start in range(0, n_samples - win_len + 1, step):
        end = start + win_len
        ecg_w = ecg[start:end]
        eda_w = eda[start:end]
        label_w = labels[start:end]

        # Only keep windows that fall entirely within an annotated,
        # in-protocol segment (WESAD uses 0 for transitions we discard).
        vals, counts = np.unique(label_w, return_counts=True)
        majority_label = vals[np.argmax(counts)]
        if majority_label == 0:
            continue  # undefined/transition segment, skip

        y = 1 if majority_label == WESAD_STRESS_LABEL else 0

        hrv = _hrv_features(ecg_w, fs)
        eda_f = _eda_features(eda_w)
        if np.isnan(hrv["hr_mean"]):
            continue  # not enough beats detected in this window, skip

        row = [hrv["hr_mean"], hrv["hr_std"], hrv["rmssd"],
               eda_f["eda_mean"], eda_f["eda_max"], eda_f["eda_std"]]
        feats.append(row)
        ys.append(y)

        # Downsample the raw sequence for the CNN-LSTM input (700Hz*20s is
        # too long/heavy; decimate to 32 Hz -> 640 timesteps per window).
        target_len = 640
        ecg_seq = resample(ecg_w, target_len)
        eda_seq = resample(eda_w, target_len)
        seqs.append(np.stack([ecg_seq, eda_seq], axis=-1))

        if subject_id is not None:
            groups.append(subject_id[start])

    X = np.array(feats)
    X_seq = np.array(seqs)
    y = np.array(ys)
    groups = np.array(groups) if groups else None

    return X, X_seq, y, feature_names, groups


if __name__ == "__main__":
    # Quick smoke test using synthetic data (fast, no download needed)
    data, source = load_wesad_or_synthetic()
    X, X_seq, y, feature_names, groups = make_windows(
        data["ecg"], data["eda"], data["labels"], subject_id=data["subject_id"]
    )
    print(f"Source: {source}")
    print(f"Tabular features X: {X.shape}, sequence X_seq: {X_seq.shape}, labels y: {y.shape}")
    print(f"Class balance: {np.bincount(y)}")
    print(f"Feature names: {feature_names}")
