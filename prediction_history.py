"""
prediction_history.py
=====================
GLOF Early Warning System — Prediction History Logger
------------------------------------------------------
Every prediction made via the web form, REST API, or MQTT pipeline
is logged to a local SQLite database.  The dashboard reads this log
to render a live history table and compute running statistics.

Public API:
    log_prediction(...)          → write one prediction record
    get_recent_predictions(n)    → return last n records as list-of-dicts
    get_stats()                  → return aggregate statistics dict
    clear_history()              → wipe all records (admin utility)
"""

import sqlite3
import logging
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

HISTORY_DB = "prediction_history.db"


# ═══════════════════════════════════════════════════════════════════════════════
# Schema initialisation
# ═══════════════════════════════════════════════════════════════════════════════

def _init_db() -> None:
    """Create the predictions table if it doesn't already exist."""
    conn = sqlite3.connect(HISTORY_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP,
            lake_id          TEXT,
            latitude         REAL,
            longitude        REAL,
            risk_class       INTEGER NOT NULL,
            risk_probability REAL    NOT NULL,
            risk_percentage  TEXT,
            inference_ms     REAL,
            engine           TEXT,
            source           TEXT DEFAULT 'web'
        )
    """)
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Write
# ═══════════════════════════════════════════════════════════════════════════════

def log_prediction(
    lake_id: str,
    latitude: Optional[float],
    longitude: Optional[float],
    risk_class: int,
    risk_probability: float,
    inference_ms: float,
    engine: str,
    source: str = "web",
) -> None:
    """
    Persist one prediction record to the SQLite history database.

    Args:
        lake_id:          Parsed lake identifier string.
        latitude:         Decimal latitude of the lake (may be None).
        longitude:        Decimal longitude of the lake (may be None).
        risk_class:       0 (Safe) or 1 (GLOF Alert).
        risk_probability: Raw sigmoid probability, 0.0–1.0.
        inference_ms:     Wall-clock inference latency in milliseconds.
        engine:           Engine label — "TFLite INT8" or "Keras .h5".
        source:           Origin of the request — "web", "api", "mqtt", "batch".
    """
    _init_db()
    try:
        conn   = sqlite3.connect(HISTORY_DB)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions
              (lake_id, latitude, longitude, risk_class, risk_probability,
               risk_percentage, inference_ms, engine, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            lake_id,
            latitude,
            longitude,
            int(risk_class),
            round(float(risk_probability), 4),
            f"{risk_probability * 100:.1f}%",
            round(float(inference_ms), 1),
            engine,
            source,
        ))
        conn.commit()
        conn.close()
        log.info("[HISTORY] Logged prediction: lake=%s class=%d prob=%.4f",
                 lake_id, risk_class, risk_probability)
    except Exception as exc:
        log.error("[HISTORY] Failed to log prediction: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# Read
# ═══════════════════════════════════════════════════════════════════════════════

def get_recent_predictions(limit: int = 20) -> list:
    """
    Return the most recent `limit` prediction records as a list of dicts,
    ordered newest-first.
    """
    _init_db()
    try:
        conn   = sqlite3.connect(HISTORY_DB)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, lake_id, latitude, longitude,
                   risk_class, risk_probability, risk_percentage,
                   inference_ms, engine, source
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "id":               row[0],
                "timestamp":        row[1],
                "lake_id":          row[2],
                "latitude":         row[3],
                "longitude":        row[4],
                "risk_class":       row[5],
                "risk_probability": row[6],
                "risk_percentage":  row[7],
                "inference_ms":     row[8],
                "engine":           row[9],
                "source":           row[10],
            }
            for row in rows
        ]
    except Exception as exc:
        log.error("[HISTORY] Failed to fetch predictions: %s", exc)
        return []


def get_stats() -> dict:
    """
    Return aggregate statistics across all logged predictions.

    Keys:
        total_predictions  (int)
        total_alerts       (int)
        alert_rate         (float)  fraction of predictions that were Class 1
        avg_probability    (float)
        avg_inference_ms   (float)
    """
    _init_db()
    try:
        conn   = sqlite3.connect(HISTORY_DB)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                COUNT(*)                    AS total,
                SUM(risk_class)             AS alerts,
                AVG(risk_probability)       AS avg_prob,
                AVG(inference_ms)           AS avg_ms
            FROM predictions
        """)
        row = cursor.fetchone()
        conn.close()
        total   = row[0] or 0
        alerts  = row[1] or 0
        return {
            "total_predictions": total,
            "total_alerts":      alerts,
            "alert_rate":        round(alerts / total, 4) if total else 0.0,
            "avg_probability":   round(row[2], 4) if row[2] else 0.0,
            "avg_inference_ms":  round(row[3], 1) if row[3] else 0.0,
        }
    except Exception as exc:
        log.error("[HISTORY] Failed to compute stats: %s", exc)
        return {}


def clear_history() -> int:
    """Delete all prediction records. Returns the number of rows deleted."""
    _init_db()
    conn   = sqlite3.connect(HISTORY_DB)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    log.info("[HISTORY] Cleared %d records.", deleted)
    return deleted
