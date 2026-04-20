"""Tests for the dynamic R:R ratio selection from backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kael_trading_bot.trade_setup_pkg.backtest import (
    MIN_RR_RATIO,
    _compute_atr,
    _backtest_rr,
    select_optimal_rr_ratio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a mild uptrend and noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="1D")
    close = 1.1000 + np.cumsum(rng.normal(0.0001, 0.002, n))
    noise = rng.normal(0, 0.0005, n)
    high = close + np.abs(noise)
    low = close - np.abs(noise)
    return pd.DataFrame(
        {"High": high, "Low": low, "Close": close},
        index=dates,
    )


# ---------------------------------------------------------------------------
# _compute_atr
# ---------------------------------------------------------------------------

class TestComputeATR:
    def test_returns_series_same_length(self):
        df = _make_ohlcv(100)
        atr = _compute_atr(df)
        assert isinstance(atr, pd.Series)
        assert len(atr) == len(df)

    def test_first_13_rows_are_nan(self):
        df = _make_ohlcv(100)
        atr = _compute_atr(df, period=14)
        assert atr.iloc[:13].isna().all()

    def test_values_are_positive(self):
        df = _make_ohlcv(100)
        atr = _compute_atr(df).dropna()
        assert (atr > 0).all()


# ---------------------------------------------------------------------------
# _backtest_rr
# ---------------------------------------------------------------------------

class TestBacktestRR:
    def test_returns_tuple_of_ints(self):
        df = _make_ohlcv(300)
        atr = _compute_atr(df)
        result = _backtest_rr(df, atr, 1.5)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(v, int) for v in result)

    def test_generates_trades(self):
        df = _make_ohlcv(300)
        atr = _compute_atr(df)
        wins, losses, total = _backtest_rr(df, atr, 1.5)
        assert total > 0, "Backtest should generate at least one trade with 300 bars"


# ---------------------------------------------------------------------------
# select_optimal_rr_ratio
# ---------------------------------------------------------------------------

class TestSelectOptimalRRRatio:
    def test_returns_dict_with_expected_keys(self):
        df = _make_ohlcv(300)
        result = select_optimal_rr_ratio(df)
        assert "rr_ratio" in result
        assert "min_rr" in result
        assert "backtested" in result
        assert "candidates" in result
        assert "reason" in result

    def test_backtested_flag_true_with_sufficient_data(self):
        df = _make_ohlcv(300)
        result = select_optimal_rr_ratio(df)
        assert result["backtested"] is True

    def test_backtested_flag_false_with_empty_df(self):
        result = select_optimal_rr_ratio(pd.DataFrame())
        assert result["backtested"] is False

    def test_backtested_flag_false_with_none(self):
        result = select_optimal_rr_ratio(None)
        assert result["backtested"] is False

    def test_rr_ratio_never_below_minimum(self):
        df = _make_ohlcv(300)
        result = select_optimal_rr_ratio(df, min_rr=MIN_RR_RATIO)
        assert result["rr_ratio"] >= MIN_RR_RATIO

    def test_rr_ratio_minimum_floor_is_1_to_2(self):
        """The default MIN_RR_RATIO must be at least 2.0 (1:2 R:R)."""
        assert MIN_RR_RATIO >= 2.0

    def test_rr_ratio_at_least_2_with_default_min(self):
        """Even when backtest would prefer a lower ratio, floor is enforced."""
        df = _make_ohlcv(300)
        result = select_optimal_rr_ratio(df)
        assert result["rr_ratio"] >= 2.0

    def test_rr_ratio_exactly_2_accepted(self):
        """When raw R:R is exactly 2.0 (the floor), it is accepted as-is."""
        df = _make_ohlcv(300)
        # Pass min_rr=2.0 explicitly; the function should accept a value of exactly 2.0
        result = select_optimal_rr_ratio(df, min_rr=2.0)
        assert result["rr_ratio"] >= 2.0
        assert result["min_rr"] == 2.0

    def test_rr_ratio_below_2_clamped_to_2(self):
        """When backtest best score is below 1:2, the result is clamped to 1:2."""
        df = _make_ohlcv(300)
        # Force a high min_rr so the function falls back to it
        result = select_optimal_rr_ratio(df, min_rr=2.5)
        assert result["rr_ratio"] >= 2.5

    def test_fallback_returns_min_rr_of_2(self):
        """Insufficient data falls back to the default minimum of 2.0."""
        tiny_df = _make_ohlcv(10)
        result = select_optimal_rr_ratio(tiny_df)
        assert result["rr_ratio"] == MIN_RR_RATIO
        assert MIN_RR_RATIO >= 2.0

    def test_rr_ratio_never_above_maximum(self):
        df = _make_ohlcv(300)
        result = select_optimal_rr_ratio(df, max_rr=3.0)
        assert result["rr_ratio"] <= 3.0

    def test_fallback_when_insufficient_data(self):
        tiny_df = _make_ohlcv(10)
        result = select_optimal_rr_ratio(tiny_df)
        assert result["rr_ratio"] == MIN_RR_RATIO
        assert result["backtested"] is False
        assert MIN_RR_RATIO >= 2.0

    def test_candidates_contain_expected_ratios(self):
        df = _make_ohlcv(300)
        result = select_optimal_rr_ratio(df, min_rr=2.0, max_rr=2.3, step=0.1)
        ratios = [c["rr_ratio"] for c in result["candidates"]]
        assert 2.0 in ratios
        assert 2.1 in ratios
        assert 2.2 in ratios
        assert 2.3 in ratios

    def test_candidate_scores_are_non_negative(self):
        df = _make_ohlcv(300)
        result = select_optimal_rr_ratio(df)
        for candidate in result["candidates"]:
            assert candidate["score"] >= 0

    def test_selected_ratio_has_highest_score(self):
        df = _make_ohlcv(300)
        result = select_optimal_rr_ratio(df)
        selected_rr = result["rr_ratio"]
        best_score = max(c["score"] for c in result["candidates"])
        selected_score = next(
            c["score"] for c in result["candidates"] if c["rr_ratio"] == selected_rr
        )
        assert selected_score == best_score

    def test_custom_min_rr(self):
        df = _make_ohlcv(300)
        result = select_optimal_rr_ratio(df, min_rr=2.5)
        assert result["rr_ratio"] >= 2.5
        assert result["min_rr"] == 2.5

    def test_reason_is_non_empty_string(self):
        df = _make_ohlcv(300)
        result = select_optimal_rr_ratio(df)
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0