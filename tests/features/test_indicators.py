"""Tests for kael_trading_bot.features.indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kael_trading_bot.features.indicators import (
    add_atr,
    add_bollinger_bands,
    add_ema,
    add_macd,
    add_rsi,
    add_sma,
)


@pytest.fixture()
def ohlcv() -> pd.DataFrame:
    """A small synthetic OHLCV DataFrame with a DatetimeIndex."""
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.2,
            "high": close + abs(np.random.randn(n) * 0.5),
            "low": close - abs(np.random.randn(n) * 0.5),
            "close": close,
            "volume": np.random.randint(100, 1000, size=n),
        },
        index=dates,
    )


# ── SMA ──────────────────────────────────────────────────────────────


class TestAddSMA:
    def test_default_periods(self, ohlcv: pd.DataFrame) -> None:
        result = add_sma(ohlcv)
        for p in (10, 20, 50):
            assert f"sma_{p}" in result.columns

    def test_custom_periods(self, ohlcv: pd.DataFrame) -> None:
        result = add_sma(ohlcv, periods=(5,))
        assert "sma_5" in result.columns
        assert "sma_10" not in result.columns

    def test_does_not_mutate_input(self, ohlcv: pd.DataFrame) -> None:
        original_cols = set(ohlcv.columns)
        add_sma(ohlcv)
        assert set(ohlcv.columns) == original_cols

    def test_nan_in_warmup(self, ohlcv: pd.DataFrame) -> None:
        result = add_sma(ohlcv, periods=(10,))
        assert result["sma_10"].iloc[:9].isna().all()
        assert not result["sma_10"].iloc[9:].isna().any()

    def test_sma_values(self, ohlcv: pd.DataFrame) -> None:
        result = add_sma(ohlcv, periods=(3,))
        expected = ohlcv["close"].rolling(3).mean()
        pd.testing.assert_series_equal(result["sma_3"], expected)


# ── EMA ──────────────────────────────────────────────────────────────


class TestAddEMA:
    def test_default_periods(self, ohlcv: pd.DataFrame) -> None:
        result = add_ema(ohlcv)
        for p in (9, 21):
            assert f"ema_{p}" in result.columns

    def test_ema_values(self, ohlcv: pd.DataFrame) -> None:
        result = add_ema(ohlcv, periods=(5,))
        expected = ohlcv["close"].ewm(span=5, adjust=False).mean()
        pd.testing.assert_series_equal(result["ema_5"], expected)


# ── RSI ──────────────────────────────────────────────────────────────


class TestAddRSI:
    def test_default_column(self, ohlcv: pd.DataFrame) -> None:
        result = add_rsi(ohlcv)
        assert "rsi_14" in result.columns

    def test_rsi_bounded(self, ohlcv: pd.DataFrame) -> None:
        result = add_rsi(ohlcv, period=14)
        rsi = result["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_custom_period(self, ohlcv: pd.DataFrame) -> None:
        result = add_rsi(ohlcv, period=5)
        assert "rsi_5" in result.columns


# ── MACD ─────────────────────────────────────────────────────────────


class TestAddMACD:
    def test_columns(self, ohlcv: pd.DataFrame) -> None:
        result = add_macd(ohlcv)
        for suffix in ("line", "signal", "hist"):
            assert f"macd_{suffix}" in result.columns

    def test_histogram_is_difference(self, ohlcv: pd.DataFrame) -> None:
        result = add_macd(ohlcv)
        expected_hist = result["macd_line"] - result["macd_signal"]
        pd.testing.assert_series_equal(result["macd_hist"], expected_hist)

    def test_custom_prefix(self, ohlcv: pd.DataFrame) -> None:
        result = add_macd(ohlcv, prefix="my_macd")
        assert "my_macd_line" in result.columns


# ── Bollinger Bands ──────────────────────────────────────────────────


class TestAddBollingerBands:
    def test_columns(self, ohlcv: pd.DataFrame) -> None:
        result = add_bollinger_bands(ohlcv)
        for suffix in ("upper", "middle", "lower", "bw", "pct_b"):
            assert f"bb_{suffix}" in result.columns

    def test_upper_greater_than_lower(self, ohlcv: pd.DataFrame) -> None:
        result = add_bollinger_bands(ohlcv)
        valid = result.dropna(subset=["bb_upper", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_bandwidth_nonnegative(self, ohlcv: pd.DataFrame) -> None:
        result = add_bollinger_bands(ohlcv)
        assert (result["bb_bw"].dropna() >= 0).all()

    def test_pct_b_within_range(self, ohlcv: pd.DataFrame) -> None:
        result = add_bollinger_bands(ohlcv)
        pct_b = result["bb_pct_b"].dropna()
        # %B can be outside [0, 1] when price exceeds bands
        # but should be finite
        assert np.isfinite(pct_b).all()


# ── ATR ──────────────────────────────────────────────────────────────


class TestAddATR:
    def test_column(self, ohlcv: pd.DataFrame) -> None:
        result = add_atr(ohlcv)
        assert "atr_14" in result.columns

    def test_atr_positive(self, ohlcv: pd.DataFrame) -> None:
        result = add_atr(ohlcv, period=14)
        atr = result["atr_14"].dropna()
        assert (atr >= 0).all()

    def test_custom_period(self, ohlcv: pd.DataFrame) -> None:
        result = add_atr(ohlcv, period=5)
        assert "atr_5" in result.columns
