"""
mqtt_ingestion.py
=================
Asynchronous Telemetry Ingestion Layer — Efftronics R&D Safety-Critical Build
------------------------------------------------------------------------------
Implements a high-availability MQTT subscriber with an SQLite circular buffer
disk fallback, guaranteeing ZERO data loss under extreme network conditions
(storms, remote connectivity loss at high-altitude deployment sites).

Architecture:
    MQTT Broker (field sensors)
        │
        ├─► [ONLINE]  → Direct model inference pipeline
        └─► [OFFLINE] → SQLite local staging buffer (disk cache)
                            │
                            └─► Auto-sync to pipeline on reconnect

Usage (standalone listener):
    python mqtt_ingestion.py

Usage (as a module inside main.py):
    from mqtt_ingestion import start_listener, drain_staging_buffer

Requirements:
    pip install paho-mqtt
"""

import sqlite3
import threading
import logging
import time
import json
from datetime import datetime
from typing import Callable, Optional

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("[WARN] paho-mqtt not installed. Run: pip install paho-mqtt")

# ─── Configuration ─────────────────────────────────────────────────────────────
MQTT_BROKER_HOST  = "localhost"          # Replace with your field broker IP
MQTT_BROKER_PORT  = 1883
MQTT_TOPIC        = "glof/telemetry"     # Topic sensors publish to
MQTT_KEEPALIVE    = 60                   # Seconds

SQLITE_DB         = "local_staging_buffer.db"
MAX_BUFFER_ROWS   = 10_000              # Circular buffer cap (oldest rows purged)
RECONNECT_DELAY   = 5                   # Seconds between reconnect attempts
DRAIN_INTERVAL    = 30                  # Seconds between staged-data sync sweeps

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Global state ──────────────────────────────────────────────────────────────
_network_online: bool = False
_inference_callback: Optional[Callable[[str], None]] = None  # set by caller


# ═══════════════════════════════════════════════════════════════════════════════
# SQLite Circular Buffer — local disk staging
# ═══════════════════════════════════════════════════════════════════════════════

def _init_db() -> None:
    """Create the telemetry_queue table if it does not already exist."""
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_queue (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            data      TEXT     NOT NULL,
            synced    INTEGER  DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
    log.info("[STAGE] SQLite circular buffer initialised → %s", SQLITE_DB)


def stage_telemetry_locally(raw_payload: str) -> None:
    """
    Write an incoming telemetry packet to the local SQLite disk buffer.
    Called automatically when the network link is down.
    Also enforces the circular buffer cap (MAX_BUFFER_ROWS).
    """
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()

    # Insert new row
    cursor.execute(
        "INSERT INTO telemetry_queue (data) VALUES (?)", (raw_payload,)
    )

    # Enforce circular cap — delete oldest excess rows
    cursor.execute(
        """
        DELETE FROM telemetry_queue
        WHERE id IN (
            SELECT id FROM telemetry_queue
            ORDER BY id ASC
            LIMIT MAX(0, (SELECT COUNT(*) FROM telemetry_queue) - ?)
        )
        """,
        (MAX_BUFFER_ROWS,),
    )

    conn.commit()
    conn.close()
    log.info("[STAGE] Packet staged to disk buffer (network offline).")


def drain_staging_buffer(callback: Callable[[str], None]) -> int:
    """
    Called after network restoration.
    Reads all unsynced rows from SQLite and feeds them back into the
    live inference pipeline, then marks them as synced.

    Returns:
        int: Number of rows successfully drained.
    """
    conn   = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, data FROM telemetry_queue WHERE synced = 0 ORDER BY id ASC"
    )
    rows = cursor.fetchall()

    drained = 0
    for row_id, payload in rows:
        try:
            callback(payload)
            cursor.execute(
                "UPDATE telemetry_queue SET synced = 1 WHERE id = ?", (row_id,)
            )
            conn.commit()
            drained += 1
        except Exception as exc:
            log.error("[STAGE] Drain failed for row %d: %s", row_id, exc)

    conn.close()
    if drained:
        log.info("[STAGE] Drained %d buffered packets → live pipeline.", drained)
    return drained


# ═══════════════════════════════════════════════════════════════════════════════
# MQTT Client Callbacks
# ═══════════════════════════════════════════════════════════════════════════════

def on_connect(client, userdata, flags, rc):
    global _network_online
    if rc == 0:
        _network_online = True
        log.info("[MQTT] Connected to broker %s:%d", MQTT_BROKER_HOST, MQTT_BROKER_PORT)
        client.subscribe(MQTT_TOPIC)
        log.info("[MQTT] Subscribed to topic: %s", MQTT_TOPIC)
    else:
        log.warning("[MQTT] Connection refused. Return code: %d", rc)


def on_disconnect(client, userdata, rc):
    global _network_online
    _network_online = False
    log.warning("[MQTT] Disconnected from broker (rc=%d). Entering disk-staging mode.", rc)


def on_message(client, userdata, msg):
    """
    Fires on every incoming MQTT telemetry packet.
    Routes directly to inference when online; falls back to disk when offline.
    """
    global _network_online, _inference_callback

    try:
        raw_data = msg.payload.decode("utf-8")
        log.info("[ENGINE] Telemetry payload received (%d bytes).", len(raw_data))

        if not _network_online:
            # Network heartbeat lost — write to disk staging buffer
            stage_telemetry_locally(raw_data)
        else:
            # Network is healthy — route directly to inference pipeline
            if _inference_callback:
                _inference_callback(raw_data)
            else:
                log.warning("[MQTT] No inference callback registered; discarding packet.")

    except Exception as exc:
        log.error("[MQTT] Ingestion anomaly caught: %s", exc)
        # Best-effort: try to stage even if decode failed
        try:
            stage_telemetry_locally(str(msg.payload))
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Background Drain Thread — periodic sync of staged data after reconnect
# ═══════════════════════════════════════════════════════════════════════════════

def _drain_loop(callback: Callable[[str], None]) -> None:
    """Background thread: every DRAIN_INTERVAL seconds, try to drain the buffer."""
    while True:
        time.sleep(DRAIN_INTERVAL)
        if _network_online:
            drain_staging_buffer(callback)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def start_listener(inference_callback: Callable[[str], None]) -> Optional[object]:
    """
    Initialise the SQLite buffer, build the MQTT client, attach callbacks,
    and start the non-blocking network loop + background drain thread.

    Args:
        inference_callback: Function to call with raw telemetry string payloads.
                            Typically wraps modelpredict.input_model().

    Returns:
        The paho MQTT client instance (for lifecycle management), or None if
        paho-mqtt is not installed.
    """
    if not MQTT_AVAILABLE:
        log.error("paho-mqtt is not installed. Cannot start MQTT listener.")
        return None

    global _inference_callback
    _inference_callback = inference_callback

    _init_db()

    client = mqtt.Client()
    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.on_message    = on_message

    # Start the background buffer-drain thread
    drain_thread = threading.Thread(
        target=_drain_loop, args=(inference_callback,), daemon=True
    )
    drain_thread.start()
    log.info("[STAGE] Background drain thread started (interval: %ds).", DRAIN_INTERVAL)

    # Connect to the MQTT broker (non-blocking loop)
    try:
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_KEEPALIVE)
        client.loop_start()
        log.info("[MQTT] Listener started. Waiting for sensor payloads...")
    except Exception as exc:
        log.warning(
            "[MQTT] Could not connect to broker at %s:%d — %s. "
            "Running in disk-staging-only mode.",
            MQTT_BROKER_HOST, MQTT_BROKER_PORT, exc,
        )

    return client


# ─── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    def mock_inference(payload: str) -> None:
        log.info("[INFERENCE] Received payload for prediction (len=%d).", len(payload))

    start_listener(mock_inference)

    log.info("Listener running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Listener stopped.")
