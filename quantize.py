"""
quantize.py
===========
Edge Quantization Script — Efftronics R&D Compliance Build
-----------------------------------------------------------
Converts the trained Keras (.h5) BiLSTM model to a TensorFlow Lite (.tflite)
binary using post-training INT8 quantization.

Result:
    bilstm_edge_model.tflite  (~4x smaller than .h5, sub-50ms inference on MCUs)

Usage:
    python quantize.py

Requirements:
    pip install tensorflow
"""

import tensorflow as tf
import numpy as np
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ─── Paths ─────────────────────────────────────────────────────────────────────
KERAS_MODEL_PATH  = "best_bilstm_model_new_soft.h5"
TFLITE_OUTPUT     = "bilstm_edge_model.tflite"
CUSTOM_LOSS_NAME  = "focal_loss_fixed"


# ─── Custom loss — required only for model loading, not for TFLite conversion ──
def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Focal Loss used during training. Needed to deserialize .h5 weights."""
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


def quantize_model() -> None:
    """Load the Keras model and export a post-training INT8 quantized TFLite binary."""

    # ── 1. Verify source model exists ─────────────────────────────────────────
    if not os.path.exists(KERAS_MODEL_PATH):
        log.error(
            "Model file '%s' not found. Run training (smot_model.py) first.",
            KERAS_MODEL_PATH,
        )
        raise FileNotFoundError(KERAS_MODEL_PATH)

    log.info("Loading Keras model from: %s", KERAS_MODEL_PATH)
    model = tf.keras.models.load_model(
        KERAS_MODEL_PATH,
        custom_objects={CUSTOM_LOSS_NAME: focal_loss(gamma=2.0, alpha=0.25)},
        compile=False,
    )
    log.info("Model loaded. Summary:")
    model.summary(print_fn=lambda x: log.info("  %s", x))

    # ── 2. Configure the TFLite converter for INT8 optimization ───────────────
    log.info("Configuring TFLite converter with INT8 post-training quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations         = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]

    # Representative dataset — feeds random samples in the correct input shape
    # so the converter can calibrate INT8 scale + zero-point per layer.
    input_shape = model.input_shape  # e.g. (None, 1, 104)

    def representative_dataset():
        for _ in range(200):
            sample = np.random.uniform(-1.0, 1.0, (1, input_shape[1], input_shape[2]))
            yield [sample.astype(np.float32)]

    converter.representative_dataset = representative_dataset

    # ── 3. Convert ────────────────────────────────────────────────────────────
    log.info("Running conversion (this may take a few seconds)...")
    tflite_model = converter.convert()

    # ── 4. Write the binary ───────────────────────────────────────────────────
    with open(TFLITE_OUTPUT, "wb") as f:
        f.write(tflite_model)

    original_mb = os.path.getsize(KERAS_MODEL_PATH) / (1024 * 1024)
    tflite_mb   = os.path.getsize(TFLITE_OUTPUT)    / (1024 * 1024)
    ratio       = original_mb / tflite_mb

    log.info("Quantization complete!")
    log.info("  Original  (.h5)    : %.2f MB", original_mb)
    log.info("  Quantized (.tflite): %.2f MB", tflite_mb)
    log.info("  Compression ratio  : %.1fx", ratio)
    log.info("  Output saved to    : %s", TFLITE_OUTPUT)


if __name__ == "__main__":
    quantize_model()
