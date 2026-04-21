"""Tests for the prediction accuracy API endpoints.

Covers all three new endpoints:

* GET /api/v1/accuracy/predictions  — paginated predictions list
* GET /api/v1/accuracy/summary       — aggregated accuracy metrics
* GET /api/v1/accuracy/trend         — accuracy trend over time

All tests mock the ``PredictionAccuracyService`` so they exercise only
the HTTP routing, input validation, response formatting, and error
handling of the API layer — no real persistence back-end is needed.

Acceptance Criteria coverage
----------------------------
AC#1 – Paginated predictions list with filtering  →  TestAccuracyPredictions
AC#2 – Aggregated accuracy summary                 →  TestAccuracySummary
AC#3 – Accuracy trend data over time               →  TestAccuracyTrend
AC#4 – Appropriate HTTP status codes & errors      →  every 4xx/5xx test
AC#5 – Integration with accuracy service            →  mocking verifies delegation
AC#6 – All endpoints covered by tests               →  this file
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from kael_trading_bot.api.app import create_app
from kael_trading_bot.prediction_accuracy.models import (
    AccuracyStatus,
    AccuracySummary,
    AccuracyTrendPoint,
    PredictionRecord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app():
    """Create a FastAPI application configured for testing."""
    return create_app()


@pytest.fixture()
def client(app):
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture()
def sample_predictions():
    """A list of sample prediction records for mocking."""
    return [
        PredictionRecord(
            id="pred-001",
            pair="EURUSD=X",
            timeframe="1h",
            predicted_direction="UP",
            predicted_at="2024-01-15T10:00:00+00:00",
            target_price=1.1000,
            actual_price=1.1050,
            percentage_drift=0.45,
            status=AccuracyStatus.CORRECT.value,
            model_name="eurusd_x_1h",
            model_version="v20240115T100000",
            created_at="2024-01-15T10:00:00+00:00",
        ),
        PredictionRecord(
            id="pred-002",
            pair="EURUSD=X",
            timeframe="1h",
            predicted_direction="DOWN",
            predicted_at="2024-01-14T10:00:00+00:00",
            target_price=1.0950,
            actual_price=1.0980,
            percentage_drift=-0.27,
            status=AccuracyStatus.INCORRECT.value,
            model_name="eurusd_x_1h",
            model_version="v20240115T100000",
            created_at="2024-01-14T10:00:00+00:00",
        ),
        PredictionRecord(
            id="pred-003",
            pair="GBPUSD=X",
            timeframe="4h",
            predicted_direction="UP",
            predicted_at="2024-01-15T08:00:00+00:00",
            target_price=1.2700,
            actual_price=None,
            percentage_drift=None,
            status=AccuracyStatus.PENDING.value,
            model_name="gbpusd_x_4h",
            model_version="v20240115T080000",
            created_at="2024-01-15T08:00:00+00:00",
        ),
    ]


@pytest.fixture()
def mock_service(sample_predictions):
    """A mocked PredictionAccuracyService with pre-configured return values."""
    svc = MagicMock()

    # list_predictions default: paginated result
    svc.list_predictions.return_value = {
        "predictions": [p.to_dict() for p in sample_predictions[:2]],
        "total": 2,
        "page": 1,
        "page_size": 20,
        "total_pages": 1,
    }

    # get_summary default
    svc.get_summary.return_value = AccuracySummary(
        pair=None,
        timeframe=None,
        total_predictions=10,
        evaluated_predictions=8,
        correct_count=5,
        incorrect_count=3,
        pending_count=2,
        win_rate=0.625,
        avg_percentage_drift=0.35,
        best_prediction=sample_predictions[0].to_dict(),
        worst_prediction=sample_predictions[1].to_dict(),
    )

    # get_trend default
    svc.get_trend.return_value = [
        AccuracyTrendPoint(
            period="2024-W02",
            total_predictions=5,
            evaluated_predictions=4,
            correct_count=3,
            win_rate=0.75,
            avg_percentage_drift=0.30,
        ),
        AccuracyTrendPoint(
            period="2024-W03",
            total_predictions=5,
            evaluated_predictions=4,
            correct_count=2,
            win_rate=0.50,
            avg_percentage_drift=0.40,
        ),
    ]

    return svc


@pytest.fixture(autouse=True)
def inject_mock_service(mock_service):
    """Automatically inject the mock service for every test in this module."""
    with patch(
        "kael_trading_bot.api.app.get_accuracy_service",
        return_value=mock_service,
    ):
        yield mock_service


# ---------------------------------------------------------------------------
# AC#1 — GET /api/v1/accuracy/predictions
# ---------------------------------------------------------------------------


class TestAccuracyPredictions:
    """Tests for the accuracy predictions list endpoint."""

    def test_predictions_returns_200(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/predictions")
        assert resp.status_code == 200

    def test_predictions_returns_paginated_structure(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/predictions")
        data = resp.json()
        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "total_pages" in data

    def test_predictions_has_accuracy_fields(self, client, mock_service):
        """Each prediction should include percentage_drift, status, etc."""
        resp = client.get("/api/v1/accuracy/predictions")
        data = resp.json()
        if data["predictions"]:
            pred = data["predictions"][0]
            for field in (
                "id", "pair", "timeframe", "predicted_direction",
                "percentage_drift", "status",
            ):
                assert field in pred, f"Missing field '{field}' in prediction"

    def test_predictions_filter_by_pair(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/predictions?pair=EURUSD=X")
        assert resp.status_code == 200
        mock_service.list_predictions.assert_called_once()
        call_kwargs = mock_service.list_predictions.call_args
        assert call_kwargs[1]["pair"] == "EURUSD=X"

    def test_predictions_filter_by_timeframe(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/predictions?timeframe=1h")
        assert resp.status_code == 200
        mock_service.list_predictions.assert_called_once()
        call_kwargs = mock_service.list_predictions.call_args
        assert call_kwargs[1]["timeframe"] == "1h"

    def test_predictions_filter_by_status_correct(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/predictions?status=correct")
        assert resp.status_code == 200
        mock_service.list_predictions.assert_called_once()
        call_kwargs = mock_service.list_predictions.call_args
        assert call_kwargs[1]["status"] == "correct"

    def test_predictions_filter_by_status_pending(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/predictions?status=pending")
        assert resp.status_code == 200
        mock_service.list_predictions.assert_called_once()
        call_kwargs = mock_service.list_predictions.call_args
        assert call_kwargs[1]["status"] == "pending"

    def test_predictions_filter_by_status_incorrect(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/predictions?status=incorrect")
        assert resp.status_code == 200
        mock_service.list_predictions.assert_called_once()
        call_kwargs = mock_service.list_predictions.call_args
        assert call_kwargs[1]["status"] == "incorrect"

    def test_predictions_invalid_status_400(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/predictions?status=maybe")
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data
        assert "Invalid status" in data["error"]

    def test_predictions_invalid_timeframe_400(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/predictions?timeframe=10m")
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data
        assert "Invalid timeframe" in data["error"]

    def test_predictions_invalid_page_400(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/predictions?page=0")
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data
        assert "page must be" in data["error"]

    def test_predictions_invalid_page_size_400(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/predictions?page_size=0")
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data
        assert "page_size must be" in data["error"]

    def test_predictions_page_size_too_large_400(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/predictions?page_size=200")
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data
        assert "page_size must be" in data["error"]

    def test_predictions_passes_pagination(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/predictions?page=2&page_size=10")
        assert resp.status_code == 200
        call_kwargs = mock_service.list_predictions.call_args
        assert call_kwargs[1]["page"] == 2
        assert call_kwargs[1]["page_size"] == 10

    def test_predictions_combined_filters(self, client, mock_service):
        """Multiple filter parameters should all be forwarded."""
        resp = client.get(
            "/api/v1/accuracy/predictions"
            "?pair=EURUSD=X&timeframe=1h&status=correct&page=1&page_size=5"
        )
        assert resp.status_code == 200
        call_kwargs = mock_service.list_predictions.call_args
        assert call_kwargs[1]["pair"] == "EURUSD=X"
        assert call_kwargs[1]["timeframe"] == "1h"
        assert call_kwargs[1]["status"] == "correct"
        assert call_kwargs[1]["page"] == 1
        assert call_kwargs[1]["page_size"] == 5

    def test_predictions_empty_result(self, client, mock_service):
        """Service returns empty list — should still be 200."""
        mock_service.list_predictions.return_value = {
            "predictions": [],
            "total": 0,
            "page": 1,
            "page_size": 20,
            "total_pages": 0,
        }
        resp = client.get("/api/v1/accuracy/predictions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["predictions"] == []
        assert data["total"] == 0

    def test_predictions_service_error_500(self, client, mock_service):
        """When the service raises an exception → 500."""
        mock_service.list_predictions.side_effect = RuntimeError("db connection failed")
        resp = client.get("/api/v1/accuracy/predictions")
        assert resp.status_code == 500
        data = resp.json()
        assert "error" in data
        assert "Internal error" in data["error"]


# ---------------------------------------------------------------------------
# AC#2 — GET /api/v1/accuracy/summary
# ---------------------------------------------------------------------------


class TestAccuracySummary:
    """Tests for the accuracy summary endpoint."""

    def test_summary_returns_200(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/summary")
        assert resp.status_code == 200

    def test_summary_has_required_fields(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/summary")
        data = resp.json()
        required_fields = (
            "total_predictions", "evaluated_predictions",
            "correct_count", "incorrect_count", "pending_count",
            "win_rate", "avg_percentage_drift",
            "best_prediction", "worst_prediction",
        )
        for field in required_fields:
            assert field in data, f"Missing field '{field}' in summary"

    def test_summary_overall(self, client, mock_service):
        """Without filters, returns overall summary."""
        resp = client.get("/api/v1/accuracy/summary")
        assert resp.status_code == 200
        mock_service.get_summary.assert_called_once_with(
            pair=None, timeframe=None,
        )

    def test_summary_filter_by_pair(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/summary?pair=EURUSD=X")
        assert resp.status_code == 200
        mock_service.get_summary.assert_called_once_with(
            pair="EURUSD=X", timeframe=None,
        )

    def test_summary_filter_by_timeframe(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/summary?timeframe=4h")
        assert resp.status_code == 200
        mock_service.get_summary.assert_called_once_with(
            pair=None, timeframe="4h",
        )

    def test_summary_filter_by_pair_and_timeframe(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/summary?pair=GBPUSD=X&timeframe=1h")
        assert resp.status_code == 200
        mock_service.get_summary.assert_called_once_with(
            pair="GBPUSD=X", timeframe="1h",
        )

    def test_summary_invalid_timeframe_400(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/summary?timeframe=10m")
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data
        assert "Invalid timeframe" in data["error"]

    def test_summary_empty_data(self, client, mock_service):
        """When no data exists, summary should have zero counts."""
        mock_service.get_summary.return_value = AccuracySummary()
        resp = client.get("/api/v1/accuracy/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_predictions"] == 0
        assert data["win_rate"] is None
        assert data["avg_percentage_drift"] is None

    def test_summary_service_error_500(self, client, mock_service):
        mock_service.get_summary.side_effect = RuntimeError("db error")
        resp = client.get("/api/v1/accuracy/summary")
        assert resp.status_code == 500
        data = resp.json()
        assert "error" in data
        assert "Internal error" in data["error"]


# ---------------------------------------------------------------------------
# AC#3 — GET /api/v1/accuracy/trend
# ---------------------------------------------------------------------------


class TestAccuracyTrend:
    """Tests for the accuracy trend endpoint."""

    def test_trend_returns_200(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/trend")
        assert resp.status_code == 200

    def test_trend_has_required_structure(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/trend")
        data = resp.json()
        assert "trend" in data
        assert isinstance(data["trend"], list)
        assert "count" in data
        assert "period" in data

    def test_trend_each_point_has_fields(self, client, mock_service):
        """Each trend point should have period, win_rate, etc."""
        resp = client.get("/api/v1/accuracy/trend")
        data = resp.json()
        if data["trend"]:
            point = data["trend"][0]
            for field in (
                "period", "total_predictions", "evaluated_predictions",
                "correct_count", "win_rate", "avg_percentage_drift",
            ):
                assert field in point, f"Missing field '{field}' in trend point"

    def test_trend_default_period_is_week(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/trend")
        assert resp.status_code == 200
        data = resp.json()
        assert data["period"] == "week"
        mock_service.get_trend.assert_called_once_with(
            pair=None, timeframe=None, period="week",
        )

    def test_trend_day_period(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/trend?period=day")
        assert resp.status_code == 200
        data = resp.json()
        assert data["period"] == "day"
        mock_service.get_trend.assert_called_once_with(
            pair=None, timeframe=None, period="day",
        )

    def test_trend_filter_by_pair(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/trend?pair=EURUSD=X")
        assert resp.status_code == 200
        mock_service.get_trend.assert_called_once_with(
            pair="EURUSD=X", timeframe=None, period="week",
        )

    def test_trend_filter_by_timeframe(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/trend?timeframe=1h")
        assert resp.status_code == 200
        mock_service.get_trend.assert_called_once_with(
            pair=None, timeframe="1h", period="week",
        )

    def test_trend_combined_filters(self, client, mock_service):
        resp = client.get(
            "/api/v1/accuracy/trend?pair=GBPUSD=X&timeframe=4h&period=day"
        )
        assert resp.status_code == 200
        mock_service.get_trend.assert_called_once_with(
            pair="GBPUSD=X", timeframe="4h", period="day",
        )

    def test_trend_invalid_period_400(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/trend?period=month")
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data
        assert "Invalid period" in data["error"]

    def test_trend_invalid_timeframe_400(self, client, mock_service):
        resp = client.get("/api/v1/accuracy/trend?timeframe=10m")
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data
        assert "Invalid timeframe" in data["error"]

    def test_trend_empty_result(self, client, mock_service):
        """When no trend data exists → 200 with empty trend list."""
        mock_service.get_trend.return_value = []
        resp = client.get("/api/v1/accuracy/trend")
        assert resp.status_code == 200
        data = resp.json()
        assert data["trend"] == []
        assert data["count"] == 0

    def test_trend_service_error_500(self, client, mock_service):
        mock_service.get_trend.side_effect = RuntimeError("db error")
        resp = client.get("/api/v1/accuracy/trend")
        assert resp.status_code == 500
        data = resp.json()
        assert "error" in data
        assert "Internal error" in data["error"]


# ---------------------------------------------------------------------------
# AC#5 — Integration with accuracy service
# ---------------------------------------------------------------------------


class TestAccuracyServiceDelegation:
    """Verify the API delegates to the accuracy service (AC#5)."""

    def test_predictions_delegates_to_service(self, client, mock_service):
        client.get("/api/v1/accuracy/predictions?pair=EURUSD=X&status=correct")
        mock_service.list_predictions.assert_called_once()
        call_kwargs = mock_service.list_predictions.call_args[1]
        assert call_kwargs["pair"] == "EURUSD=X"
        assert call_kwargs["status"] == "correct"

    def test_summary_delegates_to_service(self, client, mock_service):
        client.get("/api/v1/accuracy/summary?pair=GBPUSD=X&timeframe=4h")
        mock_service.get_summary.assert_called_once_with(
            pair="GBPUSD=X", timeframe="4h",
        )

    def test_trend_delegates_to_service(self, client, mock_service):
        client.get("/api/v1/accuracy/trend?pair=EURUSD=X&period=day")
        mock_service.get_trend.assert_called_once_with(
            pair="EURUSD=X", timeframe=None, period="day",
        )

    def test_no_accuracy_logic_in_api(self, client, mock_service):
        """Verify the API doesn't compute accuracy itself — just forwards."""
        # Even with weird input values, the service receives them unchanged
        resp = client.get("/api/v1/accuracy/predictions?page=3&page_size=50")
        assert resp.status_code == 200
        call_kwargs = mock_service.list_predictions.call_args[1]
        assert call_kwargs["page"] == 3
        assert call_kwargs["page_size"] == 50
