"""
main.py
=======
GLOF Early Warning System — Flask Web Application
Efftronics R&D Production Build — Rev 2.0
------------------------------------------
Routes:
    GET  /          → Dashboard with Plotly charts and alert feed
    POST /          → Run inference on submitted telemetry string
    GET  /about     → About page
    GET  /services  → Services page
    GET  /contact   → Contact page
    GET  /sign      → Sign-in page

Changes from Rev 1.0:
  * Structured console logging (request timestamps, inference latency)
  * MQTT listener integration (optional, non-blocking background thread)
  * MQTT connection status broadcast to the dashboard template
"""

import logging
import os

from flask import Flask, render_template, request, jsonify
from modelpredict import input_model

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── App init ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ─── Optional: MQTT listener in background thread ──────────────────────────────
# Set ENABLE_MQTT=1 in your environment to activate the listener.
# Requires: pip install paho-mqtt
ENABLE_MQTT = os.environ.get("ENABLE_MQTT", "0") == "1"
mqtt_client = None

if ENABLE_MQTT:
    try:
        from mqtt_ingestion import start_listener

        def _on_mqtt_payload(payload: str) -> None:
            """Inference callback invoked by the MQTT listener thread."""
            result = input_model(payload)
            log.info("[MQTT→INFER] Result: %s", result)

        mqtt_client = start_listener(_on_mqtt_payload)
        log.info("[MAIN] MQTT listener started in background thread.")
    except ImportError:
        log.warning("[MAIN] mqtt_ingestion module not found; MQTT disabled.")
    except Exception as exc:
        log.warning("[MAIN] Could not start MQTT listener: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        data_input = request.form.get('input_string', '').strip()
        log.info("[REQUEST] POST / — payload length: %d chars", len(data_input))

        if not data_input:
            log.warning("[REQUEST] Empty input submitted.")
            return render_template(
                'index.html',
                title="GLOF EWS",
                error="No input provided. Please paste a telemetry feature string.",
                mqtt_active=ENABLE_MQTT,
            )

        log.info("[PIPELINE] Dispatching to inference engine...")
        results = input_model(data_input)

        risk_label = (
            "⚠️  GLOF ALERT — High Risk Detected" if results.get("message") == 1
            else "✅  Normal — Low Risk"
        )
        log.info("[RESPONSE] Prediction returned: %s", risk_label)

        return render_template(
            'index.html',
            title="GLOF EWS",
            results=results,
            risk_label=risk_label,
            mqtt_active=ENABLE_MQTT,
        )

    log.info("[REQUEST] GET /")
    return render_template('index.html', title="GLOF EWS", mqtt_active=ENABLE_MQTT)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/services')
def services():
    return render_template('services.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/sign')
def sign():
    return render_template('sign.html')


@app.route('/health')
def health():
    """Lightweight health-check endpoint for monitoring probes."""
    return jsonify({"status": "ok", "mqtt_active": ENABLE_MQTT}), 200


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  GLOF Early Warning System — Rev 2.0")
    log.info("  Efftronics R&D Safety-Critical Build")
    log.info("=" * 60)
    log.info("[MAIN] Starting Flask development server...")
    app.run(debug=True)
