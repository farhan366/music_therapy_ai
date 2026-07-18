"""
train_model.py
===============
Trains the stress classifier for the closed-loop music-therapy system.

Two models are produced:
  1. RandomForest baseline on hand-crafted statistical features
     (fast, interpretable, good fallback under compute constraints).
  2. 1D CNN-LSTM hybrid on raw windowed [ECG, EDA] sequences
     (captures spatial sensor patterns + temporal dependencies, per brief).

Run:
    python train_model.py                     # synthetic dev data (fast)
    python train_model.py --wesad data/WESAD  # real WESAD data

Outputs (into ./models):
    rf_baseline.pkl        - RandomForest + StandardScaler, joblib-pickled
    cnn_lstm_stress.h5     - Keras model weights (requires TensorFlow)
    metrics.json           - accuracy / F1 for both models
"""

import argparse
import json
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data_pipeline import load_wesad_or_synthetic, load_wesad_dataset, make_windows

RANDOM_SEED = 42


def subject_wise_split(X, y, groups, test_size=0.25, seed=RANDOM_SEED):
    """Split by subject (not by window) so the same person's data never
    leaks between train and test -- important for a defensible F1 score
    in the paper. Falls back to a random split if no groups are available
    (e.g. synthetic single-block runs).
    """
    if groups is None:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        n_test = int(len(X) * test_size)
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx], train_idx, test_idx

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], train_idx, test_idx


def train_rf_baseline(X_train, y_train, X_test, y_test):
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["baseline", "stress"])

    print("\n=== RandomForest baseline ===")
    print(f"Accuracy: {acc:.4f}  |  F1: {f1:.4f}")
    print(report)

    return clf, scaler, {"accuracy": acc, "f1": f1}


def build_cnn_lstm(input_shape, n_classes=1):
    """1D CNN-LSTM hybrid classifier.

    Requires TensorFlow/Keras. This is written against the standard
    tf.keras Functional API -- install `tensorflow` in your training
    environment to run this (not available in this sandbox).
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=input_shape)  # (timesteps, channels=[ECG, EDA])

    x = layers.Conv1D(32, kernel_size=7, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(64, kernel_size=5, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="cnn_lstm_stress_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.F1Score(threshold=0.5, name="f1")],
    )
    return model


def train_cnn_lstm(X_seq_train, y_train, X_seq_test, y_test, epochs=30, batch_size=32):
    import tensorflow as tf

    # Per-channel normalization (fit on train only)
    mean = X_seq_train.mean(axis=(0, 1), keepdims=True)
    std = X_seq_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_seq_train_n = (X_seq_train - mean) / std
    X_seq_test_n = (X_seq_test - mean) / std

    model = build_cnn_lstm(input_shape=X_seq_train.shape[1:])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss"),
    ]

    history = model.fit(
        X_seq_train_n, y_train,
        validation_data=(X_seq_test_n, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    y_prob = model.predict(X_seq_test_n).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n=== CNN-LSTM ===")
    print(f"Accuracy: {acc:.4f}  |  F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["baseline", "stress"]))

    return model, {"mean": mean, "std": std}, {"accuracy": acc, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wesad", type=str, default=None,
                         help="Path to WESAD root (e.g. data/WESAD). "
                              "If omitted, uses synthetic dev data.")
    parser.add_argument("--skip-deep", action="store_true",
                         help="Skip CNN-LSTM training (e.g. no TensorFlow installed).")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)

    if args.wesad:
        data = load_wesad_dataset(args.wesad)
        source = "real"
    else:
        data, source = load_wesad_or_synthetic()

    X, X_seq, y, feature_names, groups = make_windows(
        data["ecg"], data["eda"], data["labels"], subject_id=data["subject_id"]
    )
    print(f"Data source: {source}. Windows: {len(y)}  Class balance: {np.bincount(y)}")

    X_train, X_test, y_train, y_test, train_idx, test_idx = subject_wise_split(X, y, groups)
    X_seq_train, X_seq_test = X_seq[train_idx], X_seq[test_idx]

    metrics = {"data_source": source, "n_windows": int(len(y))}

    # --- RandomForest baseline (always runs, low compute) ---
    rf_model, scaler, rf_metrics = train_rf_baseline(X_train, y_train, X_test, y_test)
    metrics["random_forest"] = rf_metrics

    with open("models/rf_baseline.pkl", "wb") as f:
        pickle.dump({"model": rf_model, "scaler": scaler, "feature_names": feature_names}, f)
    print("Saved models/rf_baseline.pkl")

    # --- CNN-LSTM (requires TensorFlow + more compute) ---
    if not args.skip_deep:
        try:
            cnn_model, norm_stats, cnn_metrics = train_cnn_lstm(
                X_seq_train, y_train, X_seq_test, y_test, epochs=args.epochs
            )
            metrics["cnn_lstm"] = cnn_metrics
            cnn_model.save("models/cnn_lstm_stress.h5")
            with open("models/cnn_lstm_norm_stats.pkl", "wb") as f:
                pickle.dump(norm_stats, f)
            print("Saved models/cnn_lstm_stress.h5")
        except ImportError:
            print("\n[train_model] TensorFlow not installed in this environment -- "
                  "skipped CNN-LSTM training. Run with TensorFlow installed, or pass "
                  "--skip-deep to silence this. The RandomForest baseline is still "
                  "saved and fully usable by the Streamlit app.")

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved models/metrics.json:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
