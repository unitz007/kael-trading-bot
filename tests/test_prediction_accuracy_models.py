"""Tests for the prediction accuracy data models.

Covers:
* PredictionRecord serialisation / deserialisation
* AccuracySummary serialisation
* AccuracyTrendPoint serialisation
* AccuracyStatus enum values
"""

from __future__ import annotations

import pytest

from kael_trading_bot.prediction_accuracy.models import (
    AccuracyStatus,
    AccuracySummary,
    AccuracyTrendPoint,
    PredictionRecord,
)


class TestAccuracyStatus:
    """Tests for the AccuracyStatus enum."""

    def test_enum_values(self):
        assert AccuracyStatus.PENDING.value == "pending"
        assert AccuracyStatus.CORRECT.value == "correct"
        assert AccuracyStatus.INCORRECT.value == "incorrect"

    def test_enum_members(self):
        assert len(AccuracyStatus) == 3


class TestPredictionRecord:
    """Tests for the PredictionRecord dataclass."""

    def test_from_dict_round_trip(self):
        original = {
            "id": "pred-001",
            "pair": "EURUSD=X",
            "timeframe": "1h",
            "predicted_direction": "UP",
            "predicted_at": "2024-01-15T10:00:00+00:00",
            "target_price": 1.1000,
            "actual_price": 1.1050,
            "percentage_drift": 0.45,
            "status": "correct",
            "model_name": "eurusd_x_1h",
            "model_version": "v20240115T100000",
            "created_at": "2024-01-15T10:00:00+00:00",
        }
        record = PredictionRecord.from_dict(original)
        assert record.id == "pred-001"
        assert record.pair == "EURUSD=X"
        assert record.percentage_drift == 0.45
        assert record.status == "correct"

    def test_to_dict(self):
        record = PredictionRecord(
            id="pred-002",
            pair="GBPUSD=X",
            timeframe="4h",
            predicted_direction="DOWN",
            predicted_at="2024-01-14T08:00:00+00:00",
            target_price=1.2700,
            actual_price=None,
            percentage_drift=None,
            status="pending",
        )
        d = record.to_dict()
        assert d["id"] == "pred-002"
        assert d["pair"] == "GBPUSD=X"
        assert d["actual_price"] is None
        assert d["percentage_drift"] is None
        assert d["status"] == "pending"

    def test_from_dict_ignores_extra_keys(self):
        """Extra keys in the input dict should be silently ignored."""
        data = {
            "id": "pred-003",
            "pair": "USDJPY=X",
            "timeframe": "1h",
            "predicted_direction": "UP",
            "predicted_at": "2024-01-15T12:00:00+00:00",
            "status": "pending",
            "unexpected_key": "should_be_ignored",
        }
        record = PredictionRecord.from_dict(data)
        assert record.id == "pred-003"
        assert not hasattr(record, "unexpected_key")

    def test_defaults(self):
        """Check default values for optional fields."""
        record = PredictionRecord(
            id="pred-004",
            pair="AUDUSD=X",
            timeframe="15m",
            predicted_direction="FLAT",
            predicted_at="2024-01-15T14:00:00+00:00",
        )
        assert record.target_price is None
        assert record.actual_price is None
        assert record.percentage_drift is None
        assert record.status == "pending"
        assert record.model_name is None
        assert record.model_version is None
        assert record.created_at == ""


class TestAccuracySummary:
    """Tests for the AccuracySummary dataclass."""

    def test_default_values(self):
        summary = AccuracySummary()
        assert summary.total_predictions == 0
        assert summary.win_rate is None
        assert summary.avg_percentage_drift is None
        assert summary.best_prediction is None
        assert summary.worst_prediction is None

    def test_to_dict(self):
        summary = AccuracySummary(
            pair="EURUSD=X",
            timeframe="1h",
            total_predictions=10,
            evaluated_predictions=8,
            correct_count=5,
            incorrect_count=3,
            pending_count=2,
            win_rate=0.625,
            avg_percentage_drift=0.35,
        )
        d = summary.to_dict()
        assert d["pair"] == "EURUSD=X"
        assert d["win_rate"] == 0.625
        assert d["correct_count"] == 5

    def test_to_dict_with_best_worst(self):
        best = {"id": "best-pred", "percentage_drift": 0.01}
        worst = {"id": "worst-pred", "percentage_drift": 5.0}
        summary = AccuracySummary(
            total_predictions=2,
            best_prediction=best,
            worst_prediction=worst,
        )
        d = summary.to_dict()
        assert d["best_prediction"] == best
        assert d["worst_prediction"] == worst


class TestAccuracyTrendPoint:
    """Tests for the AccuracyTrendPoint dataclass."""

    def test_default_values(self):
        point = AccuracyTrendPoint(period="2024-W01")
        assert point.period == "2024-W01"
        assert point.total_predictions == 0
        assert point.win_rate is None
        assert point.avg_percentage_drift is None

    def test_to_dict(self):
        point = AccuracyTrendPoint(
            period="2024-W02",
            total_predictions=5,
            evaluated_predictions=4,
            correct_count=3,
            win_rate=0.75,
            avg_percentage_drift=0.30,
        )
        d = point.to_dict()
        assert d["period"] == "2024-W02"
        assert d["win_rate"] == 0.75
