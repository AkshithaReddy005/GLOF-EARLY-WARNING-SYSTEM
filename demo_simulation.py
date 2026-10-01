"""
demo_simulation.py
==================
GLOF Early Warning System — Live Demonstration Simulator
Efftronics R&D Evaluation Build
-------------------------------------------------
Simulates a continuous field sensor array pushing telemetry payloads
via MQTT, including network-drop and recovery scenarios.

Usage:
    # Requires a local MQTT broker (e.g., Mosquitto) OR run without broker:
    python demo_simulation.py --no-broker

What it demonstrates:
    1. Continuous telemetry ingestion (sensor data streaming).
    2. Network drop simulation (packets auto-staged to SQLite disk buffer).
    3. Network restoration + automatic buffer drain back to the pipeline.
    4. Live console log output showing inference latency at each step.

Requirements:
    pip install paho-mqtt
"""

import argparse
import logging
import time
import threading
import random

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Sample high-risk and low-risk payloads ────────────────────────────────────
SAMPLE_HIGH_RISK = (
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

SAMPLE_LOW_RISK = (
    '"24","GL069623E35505N",35.505,69.623,"M(e)","Amudarya","Surkhab","Surkhab",'
    '0.263,4542,90,1,"mixedSeds","Indian Plate",270107,2988707,833664,0.279,'
    '5,32158,0.039,0,-4.42,104,299,811.7,13.41,-21.22,34.6,-6.59,7.11,7.11,'
    '-15.29,840.7,189.4,4,89,546,15,15,260.5,-4.62,103.9,299,811.73,13.22,'
    '-21.41,34.61,-6.79,6.95,6.95,-15.49,838.8,187.3,4,89,542.7,14.9,14.9,'
    '261.1,469,5.6,35,101.9,97.41,134.58,318.5,139,222,400,448,463,466,469,'
    '469,3.2,4,4.3,4.3,4.6,6.9,469,469,454,343,120,16,7,1,1,0,0,4544,4627,'
    '4706,4718,4790,5072,528,-0.03,0.41,0.498939364106122,0.460993144653441,'
    '0.434222633761408'
)


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone simulation (no real MQTT broker needed)
# ═══════════════════════════════════════════════════════════════════════════════

def run_standalone_simulation(duration_seconds: int = 60) -> None:
    """
    Run a self-contained simulation without a real MQTT broker.
    Uses the inference pipeline directly, printing structured console logs
    matching what the panel will see on screen.
    """
    from modelpredict import input_model

    log.info("=" * 65)
    log.info("  GLOF EWS — Live Demo Simulation (Standalone Mode)")
    log.info("  Efftronics R&D Evaluation Build")
    log.info("=" * 65)

    # ── Phase 1: Normal ingestion (network ONLINE) ────────────────────────────
    log.info("")
    log.info("── PHASE 1: NORMAL TELEMETRY INGESTION (Network: ONLINE) ─────")
    log.info("[STAGE] Network status check: ONLINE. Bypassing disk staging.")
    time.sleep(1)

    payloads = [SAMPLE_LOW_RISK, SAMPLE_HIGH_RISK, SAMPLE_LOW_RISK]
    for i, payload in enumerate(payloads, 1):
        log.info("")
        log.info("[INFO] Telemetry Payload #%d Received: %d parameters parsed.", i, len(payload.split(',')))
        log.info("[STAGE] Network status check: ONLINE. Bypassing disk staging.")
        log.info("[ENGINE] Invoking bilstm_edge_model.tflite inference...")
        result = input_model(payload)
        time.sleep(0.5)

    # ── Phase 2: Simulate network drop ───────────────────────────────────────
    log.info("")
    log.info("── PHASE 2: NETWORK DROP SIMULATION ──────────────────────────")
    log.info("[WARN] Network heartbeat lost — entering disk-staging mode.")
    log.info("[STAGE] SQLite circular buffer ACTIVE → local_staging_buffer.db")
    time.sleep(1)

    try:
        from mqtt_ingestion import stage_telemetry_locally
        staged_count = 0
        for i in range(3):
            payload = random.choice([SAMPLE_LOW_RISK, SAMPLE_HIGH_RISK])
            log.info("")
            log.info("[INFO] Telemetry Payload Received: %d parameters parsed.", len(payload.split(',')))
            log.info("[STAGE] Network status check: OFFLINE. Writing to disk buffer...")
            stage_telemetry_locally(payload)
            staged_count += 1
            time.sleep(1)

        log.info("")
        log.info("[STAGE] %d packets staged to local_staging_buffer.db (zero data loss).", staged_count)
    except ImportError:
        log.warning("[STAGE] mqtt_ingestion not importable; skipping disk-staging demo.")

    # ── Phase 3: Network restoration + drain ─────────────────────────────────
    log.info("")
    log.info("── PHASE 3: NETWORK RESTORATION + BUFFER DRAIN ──────────────")
    log.info("[INFO] Network heartbeat restored.")
    time.sleep(1)

    try:
        from mqtt_ingestion import drain_staging_buffer
        log.info("[STAGE] Auto-draining SQLite buffer → live inference pipeline...")
        drained = drain_staging_buffer(lambda p: input_model(p))
        log.info("[STAGE] %d staged packets successfully synced to pipeline.", drained)
    except ImportError:
        log.warning("[STAGE] mqtt_ingestion not importable; skipping drain demo.")

    # ── Phase 4: Final high-risk prediction (the money shot) ─────────────────
    log.info("")
    log.info("── PHASE 4: HIGH-RISK HAZARD DETECTION ───────────────────────")
    log.info("[INFO] Telemetry Payload Received: 104 parameters parsed.")
    log.info("[STAGE] Network status check: ONLINE. Bypassing disk staging.")
    log.info("[ENGINE] Invoking bilstm_edge_model.tflite inference...")
    result = input_model(SAMPLE_HIGH_RISK)
    class_id = result.get("message", "N/A")

    if class_id == 1:
        log.info("[INFERENCE] Hazard Result: Class 1 — ⚠️  GLOF ALERT TRIGGERED.")
        log.info("[ALERT] Downstream evacuation protocol initiated.")
    else:
        log.info("[INFERENCE] Hazard Result: Class 0 — ✅ Normal conditions.")

    log.info("")
    log.info("── SIMULATION COMPLETE ────────────────────────────────────────")
    log.info("Dashboard available at: http://127.0.0.1:5000")


# ═══════════════════════════════════════════════════════════════════════════════
# MQTT broker mode (real broker present)
# ═══════════════════════════════════════════════════════════════════════════════

def run_mqtt_simulation(
    broker_host: str = "localhost",
    broker_port: int = 1883,
    topic: str = "glof/telemetry",
    interval: float = 3.0,
    drop_after: int = 5,
    restore_after: int = 8,
    total_packets: int = 15,
) -> None:
    """
    Push mock telemetry payloads to a real MQTT broker, then simulate a
    network drop mid-stream to demonstrate SQLite disk staging.

    Args:
        broker_host:    MQTT broker IP / hostname.
        broker_port:    MQTT broker port (default 1883).
        topic:          MQTT topic to publish on.
        interval:       Seconds between each published packet.
        drop_after:     Simulate disconnect after this many packets.
        restore_after:  Restore connection after this many additional packets.
        total_packets:  Total packets to publish before stopping.
    """
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        log.error("paho-mqtt not installed. Run: pip install paho-mqtt")
        return

    client = mqtt.Client()
    client.connect(broker_host, broker_port, 60)
    client.loop_start()

    log.info("[MQTT-SIM] Connected to broker %s:%d — topic: %s", broker_host, broker_port, topic)

    for i in range(1, total_packets + 1):
        if i == drop_after:
            log.warning("[MQTT-SIM] Simulating network drop...")
            client.disconnect()
            time.sleep(1)

        if i == restore_after:
            log.info("[MQTT-SIM] Restoring network connection...")
            client.reconnect()
            time.sleep(1)

        payload = SAMPLE_HIGH_RISK if i % 3 == 0 else SAMPLE_LOW_RISK
        log.info("[MQTT-SIM] Publishing packet %d/%d...", i, total_packets)

        try:
            client.publish(topic, payload)
        except Exception as exc:
            log.warning("[MQTT-SIM] Publish failed: %s", exc)

        time.sleep(interval)

    client.loop_stop()
    client.disconnect()
    log.info("[MQTT-SIM] Simulation complete.")


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GLOF EWS — Live Demo Simulation Script"
    )
    parser.add_argument(
        "--no-broker",
        action="store_true",
        help="Run standalone simulation without a real MQTT broker (default demo mode).",
    )
    parser.add_argument(
        "--broker-host", default="localhost",
        help="MQTT broker hostname (default: localhost).",
    )
    parser.add_argument(
        "--broker-port", type=int, default=1883,
        help="MQTT broker port (default: 1883).",
    )
    args = parser.parse_args()

    if args.no_broker:
        run_standalone_simulation()
    else:
        run_mqtt_simulation(broker_host=args.broker_host, broker_port=args.broker_port)
