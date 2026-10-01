"""
modelpredict.py
===============
GLOF Early Warning System — Inference Engine
Efftronics R&D Production Build — Rev 2.0
-------------------------------------------
Changes from Rev 1.0:
  * Dual-engine support: TFLite INT8 edge binary (primary) + Keras .h5 (fallback)
  * Dynamic rolling look-back window preprocessing (look-back = 10 time steps)
  * Structured console logging (timestamp, stage, latency metrics)
  * Explicit inference latency reporting for live demo visibility

Architecture:
    Raw CSV string
        │
        ├─ Label encoding  (label_encoders.pkl)
        ├─ Standard scaling (scaler.pkl)
        ├─ 3-D rolling tensor reshape (batch, lookback, features)
        │
        ├─► TFLite INT8 engine  ← primary (sub-50ms, edge MCU target)
        └─► Keras .h5 engine    ← fallback (cloud / development)
"""

import joblib
import logging
import time
import os
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─── Paths ─────────────────────────────────────────────────────────────────────
TFLITE_MODEL_PATH = "bilstm_edge_model.tflite"
KERAS_MODEL_PATH  = "best_bilstm_model_new_soft.h5"
SCALER_PATH       = "scaler.pkl"
ENCODERS_PATH     = "label_encoders.pkl"
FEATURES_PATH     = "feature_names.pkl"

# ─── Look-back window (rolling sequence depth) ─────────────────────────────────
DEFAULT_LOOKBACK = 10   # Number of sequential time-step snapshots per inference


# ═══════════════════════════════════════════════════════════════════════════════
# Custom Loss (required to load the .h5 fallback model)
# ═══════════════════════════════════════════════════════════════════════════════

def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Focal Loss — matches the function used during training."""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_true  = tf.convert_to_tensor(y_true, tf.float32)
        y_pred  = tf.clip_by_value(
            tf.convert_to_tensor(y_pred, tf.float32), epsilon, 1.0 - epsilon
        )
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.math.pow(1.0 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(loss, axis=1)
    return focal_loss_fixed


# ═══════════════════════════════════════════════════════════════════════════════
# Artifact Loading
# ═══════════════════════════════════════════════════════════════════════════════

log.info("[INIT] Loading preprocessing artifacts...")
scaler         = joblib.load(SCALER_PATH)
label_encoders = joblib.load(ENCODERS_PATH)
feature_names  = joblib.load(FEATURES_PATH)
log.info("[INIT] Scaler, encoders, and feature names loaded. (%d features)", len(feature_names))


# ─── TFLite interpreter (primary edge engine) ──────────────────────────────────
_tflite_interpreter: Optional[tf.lite.Interpreter] = None

if os.path.exists(TFLITE_MODEL_PATH):
    log.info("[ENGINE] TFLite INT8 edge binary found → %s", TFLITE_MODEL_PATH)
    _tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    _tflite_interpreter.allocate_tensors()
    _tflite_input_details  = _tflite_interpreter.get_input_details()
    _tflite_output_details = _tflite_interpreter.get_output_details()
    log.info("[ENGINE] TFLite interpreter initialised. Primary engine: ACTIVE.")
else:
    log.warning(
        "[ENGINE] TFLite model not found at '%s'. "
        "Falling back to Keras .h5. Run quantize.py to build the edge binary.",
        TFLITE_MODEL_PATH,
    )

# ─── Keras fallback model ──────────────────────────────────────────────────────
_keras_model = None

if _tflite_interpreter is None:
    log.info("[ENGINE] Loading Keras fallback model from: %s", KERAS_MODEL_PATH)
    with custom_object_scope({"focal_loss_fixed": focal_loss(gamma=2.0, alpha=0.25)}):
        _keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    log.info("[ENGINE] Keras model loaded. Fallback engine: ACTIVE.")


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_single_snapshot(input_str: str) -> np.ndarray:
    """
    Parse a single comma-separated feature string into a normalised 1-D numpy
    array of shape (num_features,).

    Steps:
        1. Split string by comma and strip embedded quotes.
        2. Build a DataFrame aligned to feature_names.
        3. Apply label encoding to categorical columns.
        4. Cast all values to numeric (NaN-fill any coerce failures).
        5. Reorder columns to match training order.
        6. Apply StandardScaler normalisation.

    Returns:
        np.ndarray of shape (num_features,)
    """
    input_list = input_str.replace('"', '').split(',')
    data = pd.DataFrame([input_list], columns=feature_names)

    # Encode categorical features
    for col, le in label_encoders.items():
        if col in data.columns:
            known = list(le.classes_)
            fallback = known[0]
            data[col] = data[col].apply(lambda x: x if x in known else fallback)
            data[col] = le.transform(data[col])

    # Coerce all to numeric, fill NaN with 0
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Add any missing columns as zeros and reorder
    for col in feature_names:
        if col not in data.columns:
            data[col] = 0
    data = data[feature_names]

    # Scale
    scaled = scaler.transform(data)
    return scaled.flatten()  # shape: (num_features,)


def transform_to_time_series_matrix(
    historical_snapshots: np.ndarray,
    lookback_window: int = DEFAULT_LOOKBACK,
) -> np.ndarray:
    """
    Convert a 2-D array of sequential snapshots into a 3-D rolling look-back
    tensor suitable for BiLSTM layers.

    Args:
        historical_snapshots: np.ndarray of shape (num_snapshots, num_features).
                              Each row is one time-step reading from the sensor.
        lookback_window:      Number of consecutive time steps per sample.

    Returns:
        np.ndarray of shape (num_samples, lookback_window, num_features)
        where num_samples = num_snapshots - lookback_window + 1.

    Why this matters:
        Instead of evaluating a single frozen vector, the BiLSTM tracks the
        *velocity and acceleration* of each parameter over the look-back period
        (e.g., seismic magnitude trending upward over 10 days). This dramatically
        reduces false-alarm rates compared to single-snapshot inference.
    """
    if historical_snapshots.ndim != 2:
        raise ValueError(
            f"Expected 2-D input array (snapshots, features), "
            f"got shape {historical_snapshots.shape}"
        )

    n_snapshots, n_features = historical_snapshots.shape

    if n_snapshots < lookback_window:
        # Pad with zeros at the front so we always produce at least 1 sample
        pad = np.zeros((lookback_window - n_snapshots, n_features))
        historical_snapshots = np.vstack([pad, historical_snapshots])
        n_snapshots = lookback_window

    X_sequence = []
    for i in range(n_snapshots - lookback_window + 1):
        X_sequence.append(historical_snapshots[i : i + lookback_window, :])

    return np.array(X_sequence)   # (num_samples, lookback_window, num_features)


# ═══════════════════════════════════════════════════════════════════════════════
# Inference Engines
# ═══════════════════════════════════════════════════════════════════════════════

def _infer_tflite(tensor: np.ndarray) -> int:
    """
    Run a single forward pass through the TFLite INT8 interpreter.

    Args:
        tensor: np.ndarray of shape (1, lookback, features) — float32.

    Returns:
        Predicted class integer (0 = Safe, 1 = GLOF Alert).
    """
    _tflite_interpreter.set_tensor(
        _tflite_input_details[0]['index'],
        tensor.astype(np.float32),
    )
    _tflite_interpreter.invoke()
    output = _tflite_interpreter.get_tensor(_tflite_output_details[0]['index'])
    return int((output[0][0] > 0.5))


def _infer_keras(tensor: np.ndarray) -> int:
    """
    Run a single forward pass through the Keras .h5 fallback model.

    Args:
        tensor: np.ndarray of shape (1, lookback, features).

    Returns:
        Predicted class integer (0 = Safe, 1 = GLOF Alert).
    """
    predictions = _keras_model.predict(tensor, verbose=0)
    return int((predictions > 0.5).astype(int).flatten()[0])


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def predict(input_str: str, lookback_window: int = DEFAULT_LOOKBACK) -> int:
    """
    Full inference pipeline for a single telemetry payload string.

    Single-snapshot mode (standard web form):
        The snapshot is replicated `lookback_window` times to form the minimal
        valid look-back matrix. This preserves backward-compatibility with the
        existing Flask POST endpoint.

    Multi-snapshot mode (MQTT continuous stream):
        Pass a newline-delimited block of `lookback_window` CSV rows;
        each line is parsed as one time-step snapshot.

    Args:
        input_str:      Raw CSV feature string (one or multiple rows).
        lookback_window: Number of time-steps for the rolling window tensor.

    Returns:
        int — 0 (Low Risk) or 1 (GLOF Alert).
    """
    log.info("[PIPELINE] Parsing telemetry payload...")

    # ── Parse: single row or multi-row block ──────────────────────────────────
    lines = [l.strip() for l in input_str.strip().splitlines() if l.strip()]

    if len(lines) == 1:
        # Single-snapshot: replicate to fill the look-back window
        snapshot = _parse_single_snapshot(lines[0])
        snapshots = np.tile(snapshot, (lookback_window, 1))  # (lookback, features)
    else:
        snapshots = np.array([_parse_single_snapshot(l) for l in lines])

    log.info(
        "[PIPELINE] Parsed %d snapshot(s). Building %d-step look-back tensor...",
        len(lines), lookback_window,
    )

    # ── Build rolling look-back tensor ────────────────────────────────────────
    tensor = transform_to_time_series_matrix(snapshots, lookback_window)
    # Use only the last sample for single-shot inference
    input_tensor = tensor[[-1]]   # shape: (1, lookback_window, num_features)

    log.info(
        "[PIPELINE] Input tensor shape: %s. Invoking inference engine...",
        input_tensor.shape,
    )

    # ── Engine dispatch ───────────────────────────────────────────────────────
    t_start = time.perf_counter()

    if _tflite_interpreter is not None:
        log.info("[ENGINE] Invoking bilstm_edge_model.tflite (INT8)...")
        predicted_class = _infer_tflite(input_tensor)
        engine_label = "TFLite INT8"
    else:
        log.info("[ENGINE] Invoking Keras .h5 fallback model...")
        predicted_class = _infer_keras(input_tensor)
        engine_label = "Keras .h5"

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    log.info(
        "[INFERENCE] Computation executed in %.1fms via %s. "
        "Hazard Result: Class %d (%s).",
        elapsed_ms,
        engine_label,
        predicted_class,
        "GLOF ALERT 🚨" if predicted_class == 1 else "NORMAL ✅",
    )

    return predicted_class


def input_model(inpu: str) -> dict:
    """
    Flask-facing entry point.  Returns a dict consumed by Jinja2 templates.

    Returns:
        {"message": 0}  → Low Risk / Safe
        {"message": 1}  → GLOF Alert Triggered
    """
    try:
        predicted_class = predict(inpu)
        log.info("[API] Prediction result: %d", predicted_class)
        return {"message": predicted_class}
    except ValueError as exc:
        log.error("[API] Preprocessing error: %s", exc)
        return {"message": f"Error: {exc}"}
    except Exception as exc:
        log.error("[API] Unexpected inference failure: %s", exc)
        return {"message": f"Error: {exc}"}


# ─── Inline test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SAMPLE = (
        '"43","GL093898E30128N",30.128,93.898,"M(e)","Brahmaputra","Yarlung Zangbo",'
        '"Nyang",0.095,4224,0,1,"siliciclasticSeds","Plateau",143943,15522582,'
        '4249792,0.274,168,667382,0.157,0,4187,4756,5075,5085,5424,6218,2031,'
        '2.73,0.61,0.50371642588498,0.586816961265914,0.605873675652009,71,10,'
        '33,33,31,33,62.1,70,71,71,71,71,71,71,71,71,3.5,4.5,4.7,4.61,4.9,5.3,'
        '71,71,70,64,50,2,0,0,0,0,0,1.62,69,275.9,608.71,13.12,-11.98,25.09,'
        '9.52,-4.67,10.02,-6.58,579.7,107.7,3,85.1,321.9,12.8,309.4,12.8,'
        '-1.73,69,274.5,617.01,9.93,-15.3,25.21,6.35,-9.7,6.81,-9.93,756.5,'
        '144,4.3,87.1,429,15.6,414.9,15.6'
    )
    result = input_model(SAMPLE)
    print(result)