"""
modelpredict.py
===============
GLOF Early Warning System — Inference Engine
Efftronics R&D Production Build — Rev 3.0
-------------------------------------------
Changes from Rev 2.0:
  * Inference functions now return (class, probability) tuples
  * predict() returns a rich result dict: class, probability, risk_percentage,
    risk_label, inference_ms, engine, lake_id, latitude, longitude
  * New _extract_metadata() helper parses lake identity and coordinates
    directly from the raw input string using feature_names column order
  * input_model() exposes the full dict — backwards-compatible via "message" key
  * All downstream consumers (Flask, API, batch pipeline) benefit automatically
"""

import joblib
import logging
import time
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── Paths ─────────────────────────────────────────────────────────────────────
TFLITE_MODEL_PATH = "bilstm_edge_model.tflite"
KERAS_MODEL_PATH  = "best_bilstm_model_new_soft.h5"
SCALER_PATH       = "scaler.pkl"
ENCODERS_PATH     = "label_encoders.pkl"
FEATURES_PATH     = "feature_names.pkl"

DEFAULT_LOOKBACK  = 10


# ═══════════════════════════════════════════════════════════════════════════════
# Custom Loss (required to deserialise the Keras .h5 fallback model)
# ═══════════════════════════════════════════════════════════════════════════════

def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
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
log.info("[INIT] Loaded %d features.", len(feature_names))

# ─── TFLite primary engine ────────────────────────────────────────────────────
_tflite_interpreter: Optional[tf.lite.Interpreter] = None

if os.path.exists(TFLITE_MODEL_PATH):
    log.info("[ENGINE] TFLite INT8 binary found → %s", TFLITE_MODEL_PATH)
    _tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    _tflite_interpreter.allocate_tensors()
    _tflite_input_details  = _tflite_interpreter.get_input_details()
    _tflite_output_details = _tflite_interpreter.get_output_details()
    log.info("[ENGINE] TFLite interpreter ready. Primary engine: ACTIVE.")
else:
    log.warning("[ENGINE] TFLite model not found. Run quantize.py to build it.")

# ─── Keras fallback engine ────────────────────────────────────────────────────
_keras_model = None

if _tflite_interpreter is None:
    log.info("[ENGINE] Loading Keras fallback model: %s", KERAS_MODEL_PATH)
    with custom_object_scope({"focal_loss_fixed": focal_loss(gamma=2.0, alpha=0.25)}):
        _keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    log.info("[ENGINE] Keras model loaded. Fallback engine: ACTIVE.")


# ═══════════════════════════════════════════════════════════════════════════════
# Metadata Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_metadata(raw_list: list) -> dict:
    """
    Parse lake_id, latitude, and longitude from the raw split input list,
    using the column order defined in feature_names.pkl.

    Falls back gracefully if columns are not found or values are non-numeric.

    Returns:
        dict with keys: lake_id (str), latitude (float|None), longitude (float|None)
    """
    feature_list = list(feature_names)
    lake_id = str(raw_list[0]).strip() if raw_list else "UNKNOWN"
    lat, lon = None, None

    for i, fname in enumerate(feature_list):
        if i >= len(raw_list):
            break
        fname_lower = fname.lower()
        try:
            if fname_lower in ("latitude", "lat"):
                lat = float(raw_list[i])
            elif fname_lower in ("longitude", "lon"):
                lon = float(raw_list[i])
        except (ValueError, TypeError):
            pass

    return {"lake_id": lake_id, "latitude": lat, "longitude": lon}


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_single_snapshot(input_str: str) -> np.ndarray:
    """
    Parse a comma-separated feature string → normalised 1-D numpy array.
    """
    raw_list = input_str.replace('"', '').split(',')
    data = pd.DataFrame([raw_list], columns=feature_names)

    for col, le in label_encoders.items():
        if col in data.columns:
            known    = list(le.classes_)
            fallback = known[0]
            data[col] = data[col].apply(lambda x: x if x in known else fallback)
            data[col] = le.transform(data[col])

    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

    for col in feature_names:
        if col not in data.columns:
            data[col] = 0
    data = data[feature_names]

    return scaler.transform(data).flatten()


def transform_to_time_series_matrix(
    historical_snapshots: np.ndarray,
    lookback_window: int = DEFAULT_LOOKBACK,
) -> np.ndarray:
    """
    Convert a 2-D (snapshots, features) array into a 3-D rolling look-back
    tensor of shape (num_samples, lookback_window, features).
    """
    if historical_snapshots.ndim != 2:
        raise ValueError(
            f"Expected 2-D input, got shape {historical_snapshots.shape}"
        )

    n_snapshots, n_features = historical_snapshots.shape

    if n_snapshots < lookback_window:
        pad = np.zeros((lookback_window - n_snapshots, n_features))
        historical_snapshots = np.vstack([pad, historical_snapshots])
        n_snapshots = lookback_window

    return np.array([
        historical_snapshots[i: i + lookback_window, :]
        for i in range(n_snapshots - lookback_window + 1)
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# Inference Engines  (now return (class_int, probability_float) tuples)
# ═══════════════════════════════════════════════════════════════════════════════

def _infer_tflite(tensor: np.ndarray) -> Tuple[int, float]:
    """TFLite INT8 forward pass. Returns (predicted_class, probability)."""
    _tflite_interpreter.set_tensor(
        _tflite_input_details[0]['index'], tensor.astype(np.float32)
    )
    _tflite_interpreter.invoke()
    output = _tflite_interpreter.get_tensor(_tflite_output_details[0]['index'])
    prob   = float(output[0][0])
    return int(prob > 0.5), round(prob, 4)


def _infer_keras(tensor: np.ndarray) -> Tuple[int, float]:
    """Keras .h5 forward pass. Returns (predicted_class, probability)."""
    raw  = _keras_model.predict(tensor, verbose=0)
    prob = float(raw.flatten()[0])
    return int(prob > 0.5), round(prob, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def predict(input_str: str, lookback_window: int = DEFAULT_LOOKBACK) -> dict:
    """
    Full inference pipeline.

    Returns a rich result dict:
        class            (int)   0 = Safe, 1 = GLOF Alert
        message          (int)   same as class — backward-compatible key
        probability      (float) raw sigmoid output, 0.0–1.0
        risk_percentage  (str)   e.g. "87.3%"
        risk_label       (str)   human-readable verdict
        inference_ms     (float) wall-clock latency in milliseconds
        engine           (str)   "TFLite INT8" | "Keras .h5"
        lake_id          (str)   parsed from first feature column
        latitude         (float|None)
        longitude        (float|None)
    """
    log.info("[PIPELINE] Parsing telemetry payload...")
    lines = [l.strip() for l in input_str.strip().splitlines() if l.strip()]

    # Extract location metadata before encoding transforms the values
    raw_first = lines[0].replace('"', '').split(',')
    metadata  = _extract_metadata(raw_first)

    if len(lines) == 1:
        snapshot  = _parse_single_snapshot(lines[0])
        snapshots = np.tile(snapshot, (lookback_window, 1))
    else:
        snapshots = np.array([_parse_single_snapshot(l) for l in lines])

    log.info("[PIPELINE] Parsed %d snapshot(s). Building %d-step look-back tensor...",
             len(lines), lookback_window)

    tensor       = transform_to_time_series_matrix(snapshots, lookback_window)
    input_tensor = tensor[[-1]]   # shape: (1, lookback_window, num_features)

    log.info("[PIPELINE] Tensor shape: %s. Invoking engine...", input_tensor.shape)

    t_start = time.perf_counter()

    if _tflite_interpreter is not None:
        log.info("[ENGINE] Invoking bilstm_edge_model.tflite (INT8)...")
        predicted_class, probability = _infer_tflite(input_tensor)
        engine_label = "TFLite INT8"
    else:
        log.info("[ENGINE] Invoking Keras .h5 fallback...")
        predicted_class, probability = _infer_keras(input_tensor)
        engine_label = "Keras .h5"

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    risk_label = (
        "⚠️ GLOF ALERT — High Risk Detected"
        if predicted_class == 1
        else "✅ Normal — Low Risk"
    )

    log.info(
        "[INFERENCE] %.1fms | %s | Class %d | Probability %.4f (%.1f%%) | %s",
        elapsed_ms, engine_label, predicted_class,
        probability, probability * 100, risk_label,
    )

    return {
        # Core prediction
        "class":           predicted_class,
        "message":         predicted_class,   # backward-compatible key
        "probability":     probability,
        "risk_percentage": f"{probability * 100:.1f}%",
        "risk_label":      risk_label,
        # Performance
        "inference_ms":    round(elapsed_ms, 1),
        "engine":          engine_label,
        # Location
        "lake_id":         metadata["lake_id"],
        "latitude":        metadata["latitude"],
        "longitude":       metadata["longitude"],
    }


def input_model(inpu: str) -> dict:
    """
    Flask & MQTT-facing entry point.
    Returns the full predict() dict on success,
    or an error dict on failure (class = -1).
    """
    try:
        return predict(inpu)
    except ValueError as exc:
        log.error("[API] Preprocessing error: %s", exc)
        return {"class": -1, "message": f"Error: {exc}", "probability": 0.0,
                "risk_percentage": "N/A", "risk_label": "Error", "inference_ms": 0.0,
                "engine": "none", "lake_id": "UNKNOWN", "latitude": None, "longitude": None}
    except Exception as exc:
        log.error("[API] Unexpected inference failure: %s", exc)
        return {"class": -1, "message": f"Error: {exc}", "probability": 0.0,
                "risk_percentage": "N/A", "risk_label": "Error", "inference_ms": 0.0,
                "engine": "none", "lake_id": "UNKNOWN", "latitude": None, "longitude": None}


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
    import json
    print(json.dumps(input_model(SAMPLE), indent=2))