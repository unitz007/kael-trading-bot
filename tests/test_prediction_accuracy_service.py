"""Tests for the prediction accuracy service interface.

Covers:
* StubPredictionAccuracyService returns correct empty shapes
* set_accuracy_service / get_accuracy_service dependency injection
* Abstract method enforcement
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kael_trading_bot.prediction_accuracy.service import (
    PredictionAccuracyService,
    StubPredictionAccuracyService,
    get_accuracy_service,
    set_accuracy_service,
)


class TestStubPredictionAccuracyService:
    """Tests for the default stub service."""

    def test_list_predictions_returns_empty(self):
        svc = StubPredictionAccuracyService()
        result = svc.list_predictions()
        assert result["predictions"] == []
        assert result["total"] == 0
        assert result["page"] == 1
        assert result["page_size"] == 20
        assert result["total_pages"] == 0

    def test_list_predictions_passes_params(self):
        svc = StubPredictionAccuracyService()
        result = svc.list_predictions(
            pair="EURUSD=X",
            timeframe="1h",
            status="correct",
            page=2,
            page_size=10,
        )
        # Stub ignores params but should still return valid structure
        assert "predictions" in result
        assert result["page"] == 2
        assert result["page_size"] == 10

    def test_get_summary_returns_empty(self):
        svc = StubPredictionAccuracyService()
        summary = svc.get_summary()
        assert summary.total_predictions == 0
        assert summary.win_rate is None

    def test_get_summary_passes_params(self):
        svc = StubPredictionAccuracyService()
        summary = svc.get_summary(pair="EURUSD=X", timeframe="4h")
        assert summary.pair == "EURUSD=X"
        assert summary.timeframe == "4h"

    def test_get_trend_returns_empty(self):
        svc = StubPredictionAccuracyService()
        trend = svc.get_trend()
        assert trend == []

    def test_get_trend_passes_params(self):
        svc = StubPredictionAccuracyService()
        trend = svc.get_trend(pair="GBPUSD=X", period="day")
        assert isinstance(trend, list)


class TestServiceInjection:
    """Tests for the get/set_accuracy_service dependency injection."""

    def test_get_returns_stub_by_default(self):
        """When no service is set, get_accuracy_service returns a stub."""
        # Reset the module-level singleton
        import kael_trading_bot.prediction_accuracy.service as svc_module
        svc_module._service = None

        svc = get_accuracy_service()
        assert isinstance(svc, StubPredictionAccuracyService)

    def test_set_and_get(self):
        """set_accuracy_service injects a custom service."""
        import kael_trading_bot.prediction_accuracy.service as svc_module
        svc_module._service = None

        mock_svc = MagicMock(spec=PredictionAccuracyService)
        set_accuracy_service(mock_svc)
        assert get_accuracy_service() is mock_svc

        # Clean up
        svc_module._service = None

    def test_cannot_instantiate_abstract(self):
        """PredictionAccuracyService cannot be directly instantiated."""
        with pytest.raises(TypeError):
            PredictionAccuracyService()
