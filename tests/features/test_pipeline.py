"""Tests for kael_trading_bot.features.pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kael_trading_bot.features.pipeline import (
    FeatureConfig,
    _validate_input,
    build_feature_matrix,
)


@pytest.fixture()
def ohlcv() -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with enough rows for all indicators."""
    np.random.seed(42)
    n = 100
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


# ── Input validation ─────────────────────────────────────────────────


class TestValidateInput:
    def test_valid_input(self, ohlcv: pd.DataFrame) -> None:
        # Should not raise
        _validate_input(ohlcv)

    def test_missing_columns(self) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required OHLC columns"):
            _validate_input(df)


# ── Pipeline integration ─────────────────────────────────────────────


class TestBuildFeatureMatrix:
    def test_returns_dataframe(self, ohlcv: pd.DataFrame) -> None:
        result = build_feature_matrix(ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_no_nan_rows_when_drop_enabled(self, ohlcv: pd.DataFrame) -> None:
        config = FeatureConfig(drop_nan=True)
        result = build_feature_matrix(ohlcv, config)
        assert not result.isna().any().any(), "Feature matrix should not contain NaN"

    def test_original_columns_preserved(self, ohlcv: pd.DataFrame) -> None:
        result = build_feature_matrix(ohlcv)
        for col in ("open", "high", "low", "close", "volume"):
            assert col in result.columns

    def test_indicator_columns_present(self, ohlcv: pd.DataFrame) -> None:
        result = build_feature_matrix(ohlcv)
        # Spot-check a few expected columns
        for col in ("sma_10", "rsi_14", "macd_line", "atr_14", "bb_middle"):
            assert col in result.columns

    def test_time_features_present(self, ohlcv: pd.DataFrame) -> None:
        result = build_feature_matrix(ohlcv)
        assert "day_of_week" in result.columns
        assert "hour_of_day" in result.columns

    def test_time_features_disabled(self, ohlcv: pd.DataFrame) -> None:
        config = FeatureConfig(include_time_features=False)
        result = build_feature_matrix(ohlcv, config)
        assert "day_of_week" not in result.columns

    def test_target_columns_present(self, ohlcv: pd.DataFrame) -> None:
        result = build_feature_matrix(ohlcv)
        for h in (1, 5):
            assert f"future_return_{h}" in result.columns
            assert f"target_dir_{h}" in result.columns

    def test_direction_disabled(self, ohlcv: pd.DataFrame) -> None:
        config = FeatureConfig(target_direction=False)
        result = build_feature_matrix(ohlcv, config)
        assert "target_dir_1" not in result.columns
        assert "future_return_1" in result.columns

    def test_custom_horizons(self, ohlcv: pd.DataFrame) -> None:
        config = FeatureConfig(target_horizons=[3], target_direction=False)
        result = build_feature_matrix(ohlcv, config)
        assert "future_return_3" in result.columns
        assert "future_return_1" not in result.columns

    def test_drop_nan_false(self, ohlcv: pd.DataFrame) -> None:
        config = FeatureConfig(drop_nan=False)
        result = build_feature_matrix(ohlcv, config)
        # Should have NaN in target columns at the end
        assert result["future_return_1"].iloc[-1] is pd.NA or pd.isna(
            result["future_return_1"].iloc[-1]
        )

    def test_default_config(self, ohlcv: pd.DataFrame) -> None:
        # Calling without explicit config should work
        result = build_feature_matrix(ohlcv)
        assert len(result) > 0

    def test_rolling_stats_present(self, ohlcv: pd.DataFrame) -> None:
        result = build_feature_matrix(ohlcv)
        assert "close_roll_5_mean" in result.columns
        assert "close_roll_20_std" in result.columns
