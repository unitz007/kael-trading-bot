"""Tests for kael_trading_bot.features.targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kael_trading_bot.features.targets import add_future_return, add_target_direction


@pytest.fixture()
def ohlcv() -> pd.DataFrame:
    """Simple deterministic OHLCV data for easy assertions."""
    n = 10
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = np.arange(100.0, 100.0 + n, dtype=float)  # 100, 101, ..., 109
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
        },
        index=dates,
    )


# ── Future return ────────────────────────────────────────────────────


class TestAddFutureReturn:
    def test_log_return_column(self, ohlcv: pd.DataFrame) -> None:
        result = add_future_return(ohlcv, horizon=1, log=True)
        assert "future_return_1" in result.columns

    def test_simple_return_column(self, ohlcv: pd.DataFrame) -> None:
        result = add_future_return(ohlcv, horizon=1, log=False)
        assert "future_return_1" in result.columns

    def test_log_return_values(self, ohlcv: pd.DataFrame) -> None:
        result = add_future_return(ohlcv, horizon=1, log=True)
        # close = [100, 101, ..., 109]
        # log return for row 0: log(101/100)
        expected = np.log(101.0 / 100.0)
        assert abs(result["future_return_1"].iloc[0] - expected) < 1e-10

    def test_simple_return_values(self, ohlcv: pd.DataFrame) -> None:
        result = add_future_return(ohlcv, horizon=1, log=False)
        expected = (101.0 - 100.0) / 100.0
        assert abs(result["future_return_1"].iloc[0] - expected) < 1e-10

    def test_nan_at_end(self, ohlcv: pd.DataFrame) -> None:
        result = add_future_return(ohlcv, horizon=3, log=True)
        # Last 3 rows should be NaN
        assert result["future_return_3"].iloc[-1] is pd.NA or pd.isna(
            result["future_return_3"].iloc[-1]
        )

    def test_custom_horizon(self, ohlcv: pd.DataFrame) -> None:
        result = add_future_return(ohlcv, horizon=5, log=True)
        assert "future_return_5" in result.columns

    def test_does_not_mutate_input(self, ohlcv: pd.DataFrame) -> None:
        original_cols = set(ohlcv.columns)
        add_future_return(ohlcv, horizon=1)
        assert set(ohlcv.columns) == original_cols


# ── Target direction ─────────────────────────────────────────────────


class TestAddTargetDirection:
    def test_column(self, ohlcv: pd.DataFrame) -> None:
        result = add_target_direction(ohlcv, horizon=1)
        assert "target_dir_1" in result.columns

    def test_up_direction(self, ohlcv: pd.DataFrame) -> None:
        result = add_target_direction(ohlcv, horizon=1)
        # All prices go up (100->101, 101->102, etc.)
        assert (result["target_dir_1"].iloc[:-1] == 1).all()

    def test_last_row_nan(self, ohlcv: pd.DataFrame) -> None:
        result = add_target_direction(ohlcv, horizon=1)
        # Last row looks ahead to an unknown future → return = NaN → dir = 0
        # (because NaN comparison keeps the default 0)
        assert result["target_dir_1"].iloc[-1] == 0

    def test_flat_threshold(self) -> None:
        """When threshold=0 and prices are constant, direction should be 0."""
        n = 5
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        close = np.full(n, 100.0)
        df = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close},
            index=dates,
        )
        result = add_target_direction(df, horizon=1, threshold=0.0)
        assert (result["target_dir_1"].iloc[:-1] == 0).all()

    def test_down_direction(self) -> None:
        """Prices going down should produce direction=-1."""
        n = 5
        dates = pd.date_range("2024-01-01", periods=n, freq="1h")
        close = np.arange(105.0, 100.0, -1.0)  # 105, 104, 103, 102, 101
        df = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close},
            index=dates,
        )
        result = add_target_direction(df, horizon=1, threshold=0.0)
        assert (result["target_dir_1"].iloc[:-1] == -1).all()
