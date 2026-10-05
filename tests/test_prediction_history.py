"""
test_prediction_history.py
==========================
Unit tests for the prediction history SQLite logger.
"""

import os
import sys
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prediction_history import (
    log_prediction,
    get_recent_predictions,
    get_stats,
    clear_history,
    HISTORY_DB,
)


@pytest.fixture(autouse=True)
def clean_db():
    """Ensure a clean database state before each test."""
    if os.path.exists(HISTORY_DB):
        os.remove(HISTORY_DB)
    yield
    if os.path.exists(HISTORY_DB):
        os.remove(HISTORY_DB)


class TestLogPrediction:
    """Tests for writing prediction records."""

    def test_log_single_prediction(self):
        log_prediction(
            lake_id="GL093898E30128N",
            latitude=30.128,
            longitude=93.898,
            risk_class=1,
            risk_probability=0.87,
            inference_ms=14.2,
            engine="TFLite INT8",
            source="web",
        )
        records = get_recent_predictions(10)
        assert len(records) == 1
        assert records[0]["lake_id"] == "GL093898E30128N"
        assert records[0]["risk_class"] == 1

    def test_log_multiple_predictions(self):
        for i in range(5):
            log_prediction(
                lake_id=f"LAKE_{i}",
                latitude=30.0 + i,
                longitude=90.0 + i,
                risk_class=i % 2,
                risk_probability=0.1 * i,
                inference_ms=10.0 + i,
                engine="Keras .h5",
                source="api",
            )
        records = get_recent_predictions(10)
        assert len(records) == 5

    def test_log_with_none_coordinates(self):
        log_prediction(
            lake_id="UNKNOWN",
            latitude=None,
            longitude=None,
            risk_class=0,
            risk_probability=0.12,
            inference_ms=8.0,
            engine="Keras .h5",
            source="mqtt",
        )
        records = get_recent_predictions(1)
        assert len(records) == 1
        assert records[0]["latitude"] is None


class TestGetRecentPredictions:
    """Tests for reading prediction records."""

    def test_empty_database(self):
        records = get_recent_predictions(10)
        assert records == []

    def test_limit_parameter(self):
        for i in range(10):
            log_prediction(
                lake_id=f"LAKE_{i}", latitude=30.0, longitude=90.0,
                risk_class=0, risk_probability=0.1,
                inference_ms=5.0, engine="TFLite INT8", source="web",
            )
        records = get_recent_predictions(3)
        assert len(records) == 3

    def test_newest_first_ordering(self):
        log_prediction(lake_id="FIRST", latitude=0, longitude=0,
                       risk_class=0, risk_probability=0.1,
                       inference_ms=1, engine="test", source="web")
        log_prediction(lake_id="SECOND", latitude=0, longitude=0,
                       risk_class=1, risk_probability=0.9,
                       inference_ms=1, engine="test", source="web")
        records = get_recent_predictions(2)
        assert records[0]["lake_id"] == "SECOND"
        assert records[1]["lake_id"] == "FIRST"


class TestGetStats:
    """Tests for aggregate statistics."""

    def test_empty_stats(self):
        stats = get_stats()
        assert stats["total_predictions"] == 0
        assert stats["total_alerts"] == 0
        assert stats["alert_rate"] == 0.0

    def test_stats_with_data(self):
        log_prediction(lake_id="A", latitude=0, longitude=0,
                       risk_class=1, risk_probability=0.9,
                       inference_ms=10, engine="t", source="web")
        log_prediction(lake_id="B", latitude=0, longitude=0,
                       risk_class=0, risk_probability=0.2,
                       inference_ms=20, engine="t", source="web")
        stats = get_stats()
        assert stats["total_predictions"] == 2
        assert stats["total_alerts"] == 1
        assert stats["alert_rate"] == 0.5
        assert stats["avg_inference_ms"] == 15.0


class TestClearHistory:
    """Tests for clearing prediction records."""

    def test_clear_removes_all(self):
        for i in range(3):
            log_prediction(lake_id=f"L{i}", latitude=0, longitude=0,
                           risk_class=0, risk_probability=0.1,
                           inference_ms=5, engine="t", source="web")
        deleted = clear_history()
        assert deleted == 3
        assert get_recent_predictions(10) == []

    def test_clear_empty_db(self):
        deleted = clear_history()
        assert deleted == 0
