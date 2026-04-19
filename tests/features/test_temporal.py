"""Tests for kael_trading_bot.features.temporal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kael_trading_bot.features.temporal import add_rolling_stats, add_time_features


@pytest.fixture()
def ohlcv() -> pd.DataFrame:
    """A small synthetic OHLCV DataFrame with a DatetimeIndex."""
    np.random.seed(42)
    n = 30
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


# ── Time features ────────────────────────────────────────────────────


class TestAddTimeFeatures:
    def test_columns_added(self, ohlcv: pd.DataFrame) -> None:
        result = add_time_features(ohlcv)
        for col in ("day_of_week", "hour_of_day", "minute_of_hour", "is_month_start", "is_month_end"):
            assert col in result.columns

    def test_day_of_week_range(self, ohlcv: pd.DataFrame) -> None:
        result = add_time_features(ohlcv)
        dow = result["day_of_week"]
        assert (dow >= 0).all() and (dow <= 6).all()

    def test_hour_of_day_range(self, ohlcv: pd.DataFrame) -> None:
        result = add_time_features(ohlcv)
        hod = result["hour_of_day"]
        assert (hod >= 0).all() and (hod <= 23).all()

    def test_raises_on_non_datetime_index(self, ohlcv: pd.DataFrame) -> None:
        df = ohlcv.reset_index(drop=True)
        with pytest.raises(TypeError, match="DatetimeIndex"):
            add_time_features(df)

    def test_does_not_mutate_input(self, ohlcv: pd.DataFrame) -> None:
        original_cols = set(ohlcv.columns)
        add_time_features(ohlcv)
        assert set(ohlcv.columns) == original_cols


# ── Rolling statistics ───────────────────────────────────────────────


class TestAddRollingStats:
    def test_columns_added(self, ohlcv: pd.DataFrame) -> None:
        result = add_rolling_stats(ohlcv, windows=(5,))
        for suffix in ("mean", "std", "min", "max", "skew"):
            assert f"close_roll_5_{suffix}" in result.columns

    def test_custom_close_col(self, ohlcv: pd.DataFrame) -> None:
        result = add_rolling_stats(ohlcv, close_col="high", windows=(3,))
        assert "high_roll_3_mean" in result.columns

    def test_nan_in_warmup(self, ohlcv: pd.DataFrame) -> None:
        result = add_rolling_stats(ohlcv, windows=(5,))
        assert result["close_roll_5_mean"].iloc[:4].isna().all()
        assert not result["close_roll_5_mean"].iloc[4:].isna().any()

    def test_std_nonnegative(self, ohlcv: pd.DataFrame) -> None:
        result = add_rolling_stats(ohlcv, windows=(5,))
        stds = result["close_roll_5_std"].dropna()
        assert (stds >= 0).all()
