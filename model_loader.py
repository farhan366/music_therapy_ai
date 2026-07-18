"""
model_loader.py
================
Loads the trained stress classifier(s) and exposes a single, uniform
`predict_window()` call for the Streamlit app -- so app.py never has to
know whether it's talking to the RandomForest baseline or the CNN-LSTM.

Usage:
    from model_loader import StressInferenceEngine
    engine = StressInferenceEngine(prefer="cnn_lstm")  # or "rf"
    label, probability = engine.predict_window(ecg_window, eda_window)
"""

import os
import pickle

import numpy as np

from data_pipeline import _hrv_features, _eda_features, CHEST_FS


class StressInferenceEngine:
    """Unified inference wrapper.

    Tries to load the CNN-LSTM (.h5) first if prefer='cnn_lstm' and
    TensorFlow is available; otherwise falls back to the RandomForest
    baseline (.pkl). This mirrors the brief's "Alternative: RF/XGBoost
    baseline if compute constraints apply."
    """

    def __init__(self, models_dir="models", prefer="cnn_lstm", fs=CHEST_FS,
                 seq_len=640):
        self.models_dir = models_dir
        self.fs = fs
        self.seq_len = seq_len
        self.backend = None  # "cnn_lstm" or "rf"

        self.rf_model = None
        self.rf_scaler = None
        self.feature_names = None

        self.cnn_model = None
        self.cnn_norm_mean = None
        self.cnn_norm_std = None

        if prefer == "cnn_lstm":
            loaded = self._try_load_cnn_lstm()
            if not loaded:
                self._load_rf_baseline()
        else:
            self._load_rf_baseline()

    # -- loaders ------------------------------------------------------

    def _try_load_cnn_lstm(self):
        h5_path = os.path.join(self.models_dir, "cnn_lstm_stress.h5")
        norm_path = os.path.join(self.models_dir, "cnn_lstm_norm_stats.pkl")
        if not (os.path.exists(h5_path) and os.path.exists(norm_path)):
            return False
        try:
            import tensorflow as tf
            self.cnn_model = tf.keras.models.load_model(h5_path)
            with open(norm_path, "rb") as f:
                stats = pickle.load(f)
            self.cnn_norm_mean = stats["mean"]
            self.cnn_norm_std = stats["std"]
            self.backend = "cnn_lstm"
            return True
        except ImportError:
            return False

    def _load_rf_baseline(self):
        pkl_path = os.path.join(self.models_dir, "rf_baseline.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"No trained model found at '{pkl_path}'. Run train_model.py first."
            )
        with open(pkl_path, "rb") as f:
            bundle = pickle.load(f)
        self.rf_model = bundle["model"]
        self.rf_scaler = bundle["scaler"]
        self.feature_names = bundle["feature_names"]
        self.backend = "rf"

    # -- inference ------------------------------------------------------

    def predict_window(self, ecg_window, eda_window):
        """Predict on one raw [ECG, EDA] window (numpy arrays, same fs).

        Returns
        -------
        label : int, 0 = homeostasis/relaxed, 1 = acute stress
        probability : float, P(stress) in [0, 1]
        """
        ecg_window = np.asarray(ecg_window, dtype=float)
        eda_window = np.asarray(eda_window, dtype=float)

        if self.backend == "cnn_lstm":
            return self._predict_cnn_lstm(ecg_window, eda_window)
        return self._predict_rf(ecg_window, eda_window)

    def _predict_rf(self, ecg_window, eda_window):
        hrv = _hrv_features(ecg_window, self.fs)
        eda_f = _eda_features(eda_window)
        if np.isnan(hrv["hr_mean"]):
            # Not enough beats detected (e.g. buffer still filling up) --
            # default to "no stress" rather than crash the loop.
            return 0, 0.0

        row = np.array([[hrv["hr_mean"], hrv["hr_std"], hrv["rmssd"],
                          eda_f["eda_mean"], eda_f["eda_max"], eda_f["eda_std"]]])
        row_scaled = self.rf_scaler.transform(row)
        proba = self.rf_model.predict_proba(row_scaled)[0][1]
        label = int(proba >= 0.5)
        return label, float(proba)

    def _predict_cnn_lstm(self, ecg_window, eda_window):
        from scipy.signal import resample
        ecg_seq = resample(ecg_window, self.seq_len)
        eda_seq = resample(eda_window, self.seq_len)
        seq = np.stack([ecg_seq, eda_seq], axis=-1)[np.newaxis, ...]  # (1, T, 2)
        seq_norm = (seq - self.cnn_norm_mean) / self.cnn_norm_std
        proba = float(self.cnn_model.predict(seq_norm, verbose=0).ravel()[0])
        label = int(proba >= 0.5)
        return label, proba

    def predict_from_features(self, hr_mean, hr_std, rmssd, eda_mean, eda_max, eda_std):
        """Convenience path for app.py's simplified live-simulation buffers,
        which track summary stats directly rather than raw waveforms
        (see app.py's telemetry simulator). Only used with the RF backend;
        if CNN-LSTM is active this raises, since it needs raw sequences.
        """
        if self.backend != "rf":
            raise RuntimeError(
                "predict_from_features() only supports the RF backend. "
                "Use predict_window() with raw ECG/EDA arrays for CNN-LSTM."
            )
        row = np.array([[hr_mean, hr_std, rmssd, eda_mean, eda_max, eda_std]])
        row_scaled = self.rf_scaler.transform(row)
        proba = self.rf_model.predict_proba(row_scaled)[0][1]
        label = int(proba >= 0.5)
        return label, float(proba)
