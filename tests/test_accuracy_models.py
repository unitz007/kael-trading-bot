"""Tests for accuracy.models — PredictionRecord and AccuracyResult."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import pytest

from kael_trading_bot.accuracy.models import (
    AccuracyResult,
    CorrectnessStatus,
    PredictionRecord,
)


# ---------------------------------------------------------------------------
# PredictionRecord
# ---------------------------------------------------------------------------


class TestPredictionRecord:
    """Tests for the PredictionRecord data model."""

    def test_to_dict_round_trip(self):
        """Serialising and deserialising preserves all fields."""
        record = PredictionRecord(
            id=uuid.uuid4().hex,
            pair="EURUSD=X",
            timeframe="1d",
            direction="buy",
            predicted_price=1.1050,
            predicted_at="2025-01-01T00:00:00+00:00",
            horizon_at="2025-01-02T00:00:00+00:00",
            actual_price=1.1080,
            model_name="random_forest",
            model_version="1.2.0",
            generation_ts="2025-01-01T00:00:00+00:00",
        )

        as_dict = record.to_dict()
        restored = PredictionRecord.from_dict(as_dict)

        assert restored.id == record.id
        assert restored.pair == record.pair
        assert restored.timeframe == record.timeframe
        assert restored.direction == record.direction
        assert restored.predicted_price == record.predicted_price
        assert restored.predicted_at == record.predicted_at
        assert restored.horizon_at == record.horizon_at
        assert restored.actual_price == record.actual_price
        assert restored.model_name == record.model_name
        assert restored.model_version == record.model_version
        assert restored.generation_ts == record.generation_ts

    def test_from_dict_ignores_unknown_keys(self):
        """Extra keys in the dict are silently ignored."""
        data = {
            "id": "abc",
            "pair": "GBPUSD=X",
            "timeframe": "1h",
            "direction": "sell",
            "predicted_price": 1.2500,
            "predicted_at": "2025-01-01T00:00:00+00:00",
            "horizon_at": "2025-01-01T01:00:00+00:00",
            "unknown_field": "should be ignored",
        }
        record = PredictionRecord.from_dict(data)
        assert record.pair == "GBPUSD=X"
        assert not hasattr(record, "unknown_field")

    def test_defaults(self):
        """Optional fields default to empty / None."""
        record = PredictionRecord(
            pair="USDJPY=X",
            timeframe="1d",
            direction="buy",
            predicted_price=150.00,
            predicted_at="2025-01-01T00:00:00+00:00",
            horizon_at="2025-01-02T00:00:00+00:00",
        )
        assert record.id == ""
        assert record.actual_price is None
        assert record.model_name == ""
        assert record.model_version == ""
        assert record.generation_ts == ""


# ---------------------------------------------------------------------------
# AccuracyResult
# ---------------------------------------------------------------------------


class TestAccuracyResult:
    """Tests for the AccuracyResult data model."""

    def test_to_dict_includes_status_value(self):
        """Serialising an AccuracyResult converts the enum to its string value."""
        result = AccuracyResult(
            prediction_id="abc",
            pair="EURUSD=X",
            predicted_price=1.1050,
            actual_price=1.1080,
            percentage_drift=0.2710,
            predicted_direction="buy",
            actual_direction="buy",
            directional_correct=True,
            status=CorrectnessStatus.CORRECT,
        )
        d = result.to_dict()
        assert d["status"] == "correct"
        assert d["prediction_id"] == "abc"
        assert d["percentage_drift"] == 0.2710

    def test_to_dict_pending(self):
        result = AccuracyResult(
            prediction_id="xyz",
            pair="EURUSD=X",
            predicted_price=1.1050,
            actual_price=None,
            percentage_drift=None,
            predicted_direction="buy",
            actual_direction=None,
            directional_correct=None,
            status=CorrectnessStatus.PENDING,
        )
        d = result.to_dict()
        assert d["status"] == "pending"
        assert d["actual_price"] is None

    def test_to_dict_no_data(self):
        result = AccuracyResult(
            prediction_id="xyz",
            pair="EURUSD=X",
            predicted_price=1.1050,
            actual_price=None,
            percentage_drift=None,
            predicted_direction="buy",
            actual_direction=None,
            directional_correct=None,
            status=CorrectnessStatus.NO_DATA,
        )
        d = result.to_dict()
        assert d["status"] == "no_data"
