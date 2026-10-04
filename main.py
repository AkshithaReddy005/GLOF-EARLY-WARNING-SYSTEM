"""
main.py
=======
GLOF Early Warning System — Flask Web Application
Efftronics R&D Production Build — Rev 3.0
------------------------------------------
Routes:
    GET  /                  → Analytics dashboard
    POST /                  → Single-lake prediction via web form
    GET  /predict/batch     → Batch CSV upload page
    POST /predict/batch     → Process uploaded CSV, return results CSV download
    POST /api/predict       → REST JSON prediction API
    GET  /api/history       → Recent prediction history as JSON
    GET  /api/stats         → Aggregate prediction statistics as JSON
    GET  /health            → Health check (monitoring probe)
    GET  /about             → About page
    GET  /services          → Services page
    GET  /contact           → Contact page
    GET  /sign              → Sign-in page
"""

import io
import csv
import logging
import os

import pandas as pd
from flask import (
    Flask, render_template, request,
    jsonify, Response
)
from modelpredict import input_model
from prediction_history import log_prediction, get_recent_predictions, get_stats

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ─── Optional MQTT listener ───────────────────────────────────────────────────
ENABLE_MQTT = os.environ.get("ENABLE_MQTT", "0") == "1"
mqtt_client = None

if ENABLE_MQTT:
    try:
        from mqtt_ingestion import start_listener

        def _on_mqtt_payload(payload: str) -> None:
            result = input_model(payload)
            log_prediction(
                lake_id=result.get("lake_id", "MQTT"),
                latitude=result.get("latitude"),
                longitude=result.get("longitude"),
                risk_class=result.get("class", -1),
                risk_probability=result.get("probability", 0.0),
                inference_ms=result.get("inference_ms", 0.0),
                engine=result.get("engine", "unknown"),
                source="mqtt",
            )
            log.info("[MQTT→INFER] Result: class=%d prob=%.4f",
                     result.get("class", -1), result.get("probability", 0.0))

        mqtt_client = start_listener(_on_mqtt_payload)
    except Exception as exc:
        log.warning("[MAIN] MQTT listener failed to start: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper — log and return a prediction result dict
# ═══════════════════════════════════════════════════════════════════════════════

def _predict_and_log(input_str: str, source: str = "web") -> dict:
    """Run inference and persist the result to the history log."""
    result = input_model(input_str)
    if result.get("class", -1) != -1:
        try:
            log_prediction(
                lake_id=result.get("lake_id", "UNKNOWN"),
                latitude=result.get("latitude"),
                longitude=result.get("longitude"),
                risk_class=result.get("class"),
                risk_probability=result.get("probability", 0.0),
                inference_ms=result.get("inference_ms", 0.0),
                engine=result.get("engine", "unknown"),
                source=source,
            )
        except Exception as exc:
            log.warning("[MAIN] History log failed: %s", exc)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Web Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data_input = request.form.get("input_string", "").strip()
        log.info("[REQUEST] POST / — payload: %d chars", len(data_input))

        if not data_input:
            return render_template(
                "index.html", title="GLOF EWS",
                error="No input provided. Please paste a telemetry feature string.",
                mqtt_active=ENABLE_MQTT,
            )

        result = _predict_and_log(data_input, source="web")
        log.info("[RESPONSE] class=%d prob=%.4f lake=%s",
                 result.get("class", -1), result.get("probability", 0.0),
                 result.get("lake_id", "?"))

        return render_template(
            "index.html", title="GLOF EWS",
            results=result,
            mqtt_active=ENABLE_MQTT,
        )

    return render_template("index.html", title="GLOF EWS", mqtt_active=ENABLE_MQTT)


@app.route("/predict/batch", methods=["GET", "POST"])
def predict_batch():
    """
    GET  → Render the CSV upload form.
    POST → Process uploaded CSV, run inference on every row,
           return a downloadable results CSV with Risk_Class,
           Risk_Probability, and Risk_Percentage appended.
    """
    if request.method == "POST":
        if "csv_file" not in request.files or request.files["csv_file"].filename == "":
            return render_template("batch.html", error="No CSV file selected.")

        file = request.files["csv_file"]
        log.info("[BATCH] Received file: %s", file.filename)

        try:
            content  = file.stream.read().decode("utf-8")
            df_input = pd.read_csv(io.StringIO(content))

            output_rows   = []
            alert_count   = 0
            total_rows    = 0

            for _, row in df_input.iterrows():
                total_rows += 1
                # Features = all columns except last 2 (GLOF, Date)
                feature_slice = row.iloc[:-2] if len(row) > 2 else row
                input_str = ",".join(
                    f'"{v}"' if isinstance(v, str) else str(v)
                    for v in feature_slice
                )

                result = _predict_and_log(input_str, source="batch")

                # Append prediction columns to the original row
                row_out            = row.to_dict()
                row_out["Risk_Class"]       = result.get("class", -1)
                row_out["Risk_Probability"] = result.get("probability", 0.0)
                row_out["Risk_Percentage"]  = result.get("risk_percentage", "N/A")
                row_out["Risk_Label"]       = result.get("risk_label", "N/A")
                row_out["Inference_ms"]     = result.get("inference_ms", 0.0)
                row_out["Engine"]           = result.get("engine", "N/A")
                output_rows.append(row_out)

                if result.get("class") == 1:
                    alert_count += 1

            log.info("[BATCH] Processed %d rows. Alerts: %d", total_rows, alert_count)

            # Build downloadable results CSV
            if not output_rows:
                return render_template("batch.html",
                                       error="The uploaded CSV contained no processable rows.")

            out_stream = io.StringIO()
            writer     = csv.DictWriter(out_stream, fieldnames=output_rows[0].keys())
            writer.writeheader()
            writer.writerows(output_rows)
            out_stream.seek(0)

            return Response(
                out_stream.getvalue(),
                mimetype="text/csv",
                headers={
                    "Content-Disposition": (
                        f"attachment; filename=glof_batch_predictions_{total_rows}lakes.csv"
                    )
                },
            )

        except Exception as exc:
            log.error("[BATCH] Processing failed: %s", exc)
            return render_template("batch.html",
                                   error=f"Processing error: {exc}")

    return render_template("batch.html")


# ═══════════════════════════════════════════════════════════════════════════════
# REST API Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST JSON prediction endpoint.

    Accepts JSON body: {"input_string": "<csv feature string>"}
    or form-encoded:   input_string=<csv feature string>

    Returns:
        200  {"class": int, "probability": float, "risk_percentage": str,
              "risk_label": str, "inference_ms": float, "engine": str,
              "lake_id": str, "latitude": float|null, "longitude": float|null}
        400  {"error": "input_string is required"}
        500  {"error": "<message>"}
    """
    data      = request.get_json(silent=True) or {}
    input_str = data.get("input_string") or request.form.get("input_string", "").strip()

    if not input_str:
        return jsonify({"error": "input_string is required"}), 400

    log.info("[API] POST /api/predict — payload: %d chars", len(input_str))

    result = _predict_and_log(input_str, source="api")

    if result.get("class", -1) == -1:
        return jsonify({"error": result.get("message", "Inference failed")}), 500

    return jsonify(result), 200


@app.route("/api/history", methods=["GET"])
def api_history():
    """
    Return recent prediction history as JSON.

    Query params:
        limit (int, default 20) — number of records to return

    Returns:
        {"predictions": [...], "count": int}
    """
    try:
        limit   = int(request.args.get("limit", 20))
        history = get_recent_predictions(limit)
        return jsonify({"predictions": history, "count": len(history)}), 200
    except Exception as exc:
        log.error("[API] /api/history failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """
    Return aggregate prediction statistics.

    Returns:
        {"total_predictions": int, "total_alerts": int,
         "alert_rate": float, "avg_probability": float, "avg_inference_ms": float}
    """
    try:
        return jsonify(get_stats()), 200
    except Exception as exc:
        log.error("[API] /api/stats failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Lightweight health-check for uptime monitoring probes."""
    return jsonify({"status": "ok", "mqtt_active": ENABLE_MQTT}), 200


# ═══════════════════════════════════════════════════════════════════════════════
# Page Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/services")
def services():
    return render_template("services.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/sign")
def sign():
    return render_template("sign.html")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  GLOF Early Warning System — Rev 3.0")
    log.info("  Efftronics R&D Safety-Critical Build")
    log.info("  Routes: / | /predict/batch | /api/predict")
    log.info("          /api/history | /api/stats | /health")
    log.info("=" * 60)
    app.run(debug=True)
