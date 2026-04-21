"""Tests for accuracy.calculator — accuracy evaluation logic."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from unittest.mock import MagicMock, patch

from kael_trading_bot.accuracy.calculator import (
    calculate_percentage_drift,
    check_directional_correctness,
    determine_direction,
    evaluate_prediction,
    evaluate_predictions,
    get_current_status,
    fetch_actual_price,
)

from kael_trading_bot.accuracy.models import (
    AccuracyResult,
    CorrectnessStatus,
    PredictionRecord,
)


def _make_prediction(
    direction: str = "buy",
    predicted_price: float = 1.1050,
    predicted_at: str = "2025-01-01T00:00:00+00:00",
    horizon_at: str | None = None,
    **kwargs,
) -> PredictionRecord:
    if horizon_at is None:
        # Default to 1 day before now so the horizon is in the past
        horizon_at = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    return PredictionRecord(
        id=kwargs.get("id", uuid.uuid4().hex),
        pair=kwargs.get("pair", "EURUSD=X"),
        timeframe=kwargs.get("timeframe", "1d"),
        direction=direction,
        predicted_price=predicted_price,
        predicted_at=predicted_at,
        horizon_at=horizon_at,
        model_name=kwargs.get("model_name", "rf"),
        model_version=kwargs.get("model_version", "1.0"),
        generation_ts=kwargs.get("generation_ts", predicted_at),
    )


def _future_horizon_iso() -> str:
    return (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()


def _past_horizon_iso() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()


# ---------------------------------------------------------------------------
# calculate_percentage_drift
# ---------------------------------------------------------------------------


class TestCalculatePercentageDrift:
    def test_zero_drift(self):
        assert calculate_percentage_drift(100.0, 100.0) == 0.0

    def test_positive_drift(self):
        result = calculate_percentage_drift(105.0, 100.0)
        assert result == pytest.approx(5.0)

    def test_negative_drift_absolute(self):
        result = calculate_percentage_drift(95.0, 100.0)
        assert result == pytest.approx(5.0)

    def test_small_drift(self):
        result = calculate_percentage_drift(1.1050, 1.1080)
        expected = abs(1.1050 - 1.1080) / 1.1080 * 100
        assert result == pytest.approx(expected, rel=1e-6)

    def test_zero_actual_price_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            calculate_percentage_drift(1.0, 0.0)

    def test_negative_actual_price_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            calculate_percentage_drift(1.0, -50.0)


# ---------------------------------------------------------------------------
# determine_direction
# ---------------------------------------------------------------------------


class TestDetermineDirection:
    def test_price_up(self):
        assert determine_direction(1.0, 1.1) == "buy"

    def test_price_down(self):
        assert determine_direction(1.1, 1.0) == "sell"

    def test_price_flat(self):
        assert determine_direction(1.0, 1.0) == "flat"


# ---------------------------------------------------------------------------
# check_directional_correctness
# ---------------------------------------------------------------------------


class TestCheckDirectionalCorrectness:
    def test_buy_matches_buy(self):
        assert check_directional_correctness("buy", "buy") is True

    def test_sell_matches_sell(self):
        assert check_directional_correctness("sell", "sell") is True

    def test_buy_vs_sell(self):
        assert check_directional_correctness("buy", "sell") is False

    def test_sell_vs_buy(self):
        assert check_directional_correctness("sell", "buy") is False

    def test_flat_actual_not_correct_for_buy(self):
        assert check_directional_correctness("buy", "flat") is False

    def test_flat_actual_not_correct_for_sell(self):
        assert check_directional_correctness("sell", "flat") is False

    def test_flat_prediction_flat_actual(self):
        assert check_directional_correctness("flat", "flat") is True

    def test_flat_prediction_nonflat_actual(self):
        assert check_directional_correctness("flat", "buy") is False


# ---------------------------------------------------------------------------
# get_current_status
# ---------------------------------------------------------------------------


class TestGetCurrentStatus:
    def test_future_horizon(self):
        future = _future_horizon_iso()
        assert get_current_status(future) == "pending"

    def test_past_horizon(self):
        past = _past_horizon_iso()
        assert get_current_status(past) == "expired"

    def test_invalid_timestamp(self):
        assert get_current_status("not-a-date") == "expired"


# ---------------------------------------------------------------------------
# fetch_actual_price
# ---------------------------------------------------------------------------


class TestFetchActualPrice:
    @patch("kael_trading_bot.accuracy.calculator.ForexDataFetcher")
    def test_returns_close_price_from_df(self, mock_fetcher_cls):
        """When the fetcher returns a DataFrame with a matching row, return the Close price."""
        import pandas as pd

        idx = pd.date_range("2025-01-01", periods=3, freq="D")
        df = pd.DataFrame({"Close": [1.10, 1.1050, 1.11]}, index=idx)

        mock_instance = MagicMock()
        mock_instance.get.return_value = df
        mock_fetcher_cls.return_value = mock_instance

        result = fetch_actual_price("EURUSD=X", "2025-01-02T00:00:00+00:00", timeframe="1d")
        assert result == 1.1050

    @patch("kael_trading_bot.accuracy.calculator.ForexDataFetcher")
    def test_returns_none_when_df_empty(self, mock_fetcher_cls):
        import pandas as pd

        mock_instance = MagicMock()
        mock_instance.get.return_value = pd.DataFrame()
        mock_fetcher_cls.return_value = mock_instance

        result = fetch_actual_price("EURUSD=X", "2025-01-02T00:00:00+00:00")
        assert result is None

    @patch("kael_trading_bot.accuracy.calculator.ForexDataFetcher")
    def test_returns_none_on_exception(self, mock_fetcher_cls):
        mock_fetcher_cls.side_effect = Exception("network error")

        result = fetch_actual_price("EURUSD=X", "2025-01-02T00:00:00+00:00")
        assert result is None


# ---------------------------------------------------------------------------
# evaluate_prediction
# ---------------------------------------------------------------------------


class TestEvaluatePrediction:
    def test_pending_when_horizon_in_future(self):
        pred = _make_prediction(horizon_at=_future_horizon_iso())
        result = evaluate_prediction(pred, actual_price=None)
        assert result.status == CorrectnessStatus.PENDING
        assert result.actual_price is None
        assert result.percentage_drift is None
        assert result.directional_correct is None

    def test_no_data_when_horizon_passed_no_price(self):
        pred = _make_prediction(horizon_at=_past_horizon_iso())
        result = evaluate_prediction(pred, actual_price=None)
        assert result.status == CorrectnessStatus.NO_DATA
        assert result.actual_price is None

    def test_correct_when_drift_within_tolerance_and_direction_matches(self):
        pred = _make_prediction(
            direction="buy",
            predicted_price=1.1050,
            horizon_at=_past_horizon_iso(),
        )
        # actual price went up → direction matches, drift is small
        result = evaluate_prediction(
            pred,
            actual_price=1.1060,
            price_at_generation=1.1000,
            tolerance_pct=2.0,
        )
        assert result.status == CorrectnessStatus.CORRECT
        assert result.directional_correct is True
        assert result.percentage_drift is not None
        assert result.percentage_drift < 2.0

    def test_incorrect_when_drift_exceeds_tolerance(self):
        pred = _make_prediction(
            direction="buy",
            predicted_price=1.1050,
            horizon_at=_past_horizon_iso(),
        )
        result = evaluate_prediction(
            pred,
            actual_price=1.2000,
            price_at_generation=1.1000,
            tolerance_pct=2.0,
        )
        assert result.status == CorrectnessStatus.INCORRECT
        # Direction is correct but drift is too large
        assert result.directional_correct is True
        assert result.percentage_drift > 2.0

    def test_incorrect_when_direction_wrong(self):
        pred = _make_prediction(
            direction="buy",
            predicted_price=1.1050,
            horizon_at=_past_horizon_iso(),
        )
        # actual price went down → wrong direction
        result = evaluate_prediction(
            pred,
            actual_price=1.1000,
            price_at_generation=1.1050,
            tolerance_pct=2.0,
        )
        assert result.status == CorrectnessStatus.INCORRECT
        assert result.directional_correct is False

    def test_sell_direction_correct(self):
        pred = _make_prediction(
            direction="sell",
            predicted_price=1.1000,
            horizon_at=_past_horizon_iso(),
        )
        result = evaluate_prediction(
            pred,
            actual_price=1.0950,
            price_at_generation=1.1000,
            tolerance_pct=2.0,
        )
        assert result.status == CorrectnessStatus.CORRECT
        assert result.actual_direction == "sell"
        assert result.directional_correct is True

    def test_uses_predicted_price_as_gen_price_when_not_provided(self):
        pred = _make_prediction(
            direction="buy",
            predicted_price=1.1050,
            horizon_at=_past_horizon_iso(),
        )
        # No price_at_generation → uses predicted_price (1.1050)
        # actual_price 1.1100 → direction is "buy" → matches
        result = evaluate_prediction(pred, actual_price=1.1100)
        assert result.actual_direction == "buy"
        assert result.directional_correct is True

    def test_custom_tolerance(self):
        pred = _make_prediction(
            direction="buy",
            predicted_price=1.1050,
            horizon_at=_past_horizon_iso(),
        )
        # 5% drift, tolerance 3% → incorrect
        result = evaluate_prediction(
            pred,
            actual_price=1.1603,
            price_at_generation=1.1000,
            tolerance_pct=3.0,
        )
        assert result.status == CorrectnessStatus.INCORRECT
        assert result.tolerance_pct == 3.0


# ---------------------------------------------------------------------------
# evaluate_predictions (batch)
# ---------------------------------------------------------------------------


class TestEvaluatePredictions:
    def test_batch_evaluation(self):
        p1 = _make_prediction(id="p1", horizon_at=_past_horizon_iso())
        p2 = _make_prediction(id="p2", horizon_at=_future_horizon_iso())

        actual_prices = {"p1": 1.1060, "p2": None}
        gen_prices = {"p1": 1.1000, "p2": 1.1000}

        results = evaluate_predictions(
            [p1, p2],
            actual_prices=actual_prices,
            prices_at_generation=gen_prices,
        )

        assert len(results) == 2
        assert results[0].prediction_id == "p1"
        assert results[1].prediction_id == "p2"
        # p1 has past horizon + actual price → should be evaluated
        assert results[0].status in (CorrectnessStatus.CORRECT, CorrectnessStatus.INCORRECT)
        # p2 has future horizon → pending
        assert results[1].status == CorrectnessStatus.PENDING

    def test_batch_no_gen_prices(self):
        p = _make_prediction(id="p1", horizon_at=_past_horizon_iso())
        results = evaluate_predictions(
            [p],
            actual_prices={"p1": 1.1100},
        )
        assert len(results) == 1
        # Should not raise — uses predicted_price as fallback
        assert results[0].status in (CorrectnessStatus.CORRECT, CorrectnessStatus.INCORRECT)

    def test_batch_empty(self):
        assert evaluate_predictions([], {}) == []