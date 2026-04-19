"""Feature engineering pipeline.

The :func:`build_feature_matrix` function is the main entry point.  It
chains indicator computation, temporal features, target generation, and
NaN handling into a single callable so downstream consumers receive a
clean, ready-to-use DataFrame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

from kael_trading_bot.features.indicators import (
    add_atr,
    add_bollinger_bands,
    add_ema,
    add_macd,
    add_rsi,
    add_sma,
)
from kael_trading_bot.features.targets import add_future_return, add_target_direction
from kael_trading_bot.features.temporal import add_rolling_stats, add_time_features

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for the feature engineering pipeline."""

    # --- Moving averages ---
    sma_periods: tuple[int, ...] = (10, 20, 50)
    ema_periods: tuple[int, ...] = (9, 21)

    # --- RSI ---
    rsi_period: int = 14

    # --- MACD ---
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # --- Bollinger Bands ---
    bb_period: int = 20
    bb_std: float = 2.0

    # --- ATR ---
    atr_period: int = 14

    # --- Rolling stats ---
    rolling_windows: tuple[int, ...] = (5, 10, 20)

    # --- Time features ---
    include_time_features: bool = True

    # --- Targets ---
    target_horizons: Sequence[int] = field(default_factory=lambda: [1, 5])
    target_log_returns: bool = True
    target_direction: bool = True
    target_threshold: float = 0.0

    # --- NaN handling ---
    drop_nan: bool = True


def build_feature_matrix(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Transform raw OHLCV data into a clean feature matrix.

    The pipeline runs in this order:

    1. Validate input columns.
    2. Compute technical indicators (SMA, EMA, RSI, MACD, BB, ATR).
    3. Add rolling window statistics.
    4. Add time-based features (if enabled).
    5. Add target variables (future returns, direction).
    6. Drop rows with NaN (if enabled) to prevent leakage.

    Parameters
    ----------
    df:
        DataFrame with at least ``open``, ``high``, ``low``, ``close``
        columns and a ``DatetimeIndex``.
    config:
        Pipeline configuration.  Uses defaults when ``None``.

    Returns
    -------
    pd.DataFrame
        Feature matrix ready for ML model consumption.

    Raises
    ------
    ValueError
        If required columns are missing from *df*.
    TypeError
        If the DataFrame index is not a ``DatetimeIndex`` and time
        features are enabled.
    """
    if config is None:
        config = FeatureConfig()

    _validate_input(df)

    logger.info("Starting feature engineering on %d rows", len(df))

    # 1. Technical indicators
    result = add_sma(df, periods=config.sma_periods)
    result = add_ema(result, periods=config.ema_periods)
    result = add_rsi(result, period=config.rsi_period)
    result = add_macd(
        result,
        fast=config.macd_fast,
        slow=config.macd_slow,
        signal=config.macd_signal,
    )
    result = add_bollinger_bands(result, period=config.bb_period, num_std=config.bb_std)
    result = add_atr(result, period=config.atr_period)
    logger.info("Technical indicators added")

    # 2. Rolling statistics
    result = add_rolling_stats(result, windows=config.rolling_windows)
    logger.info("Rolling statistics added")

    # 3. Time features
    if config.include_time_features:
        result = add_time_features(result)
        logger.info("Time features added")

    # 4. Targets
    for h in config.target_horizons:
        result = add_future_return(
            result, horizon=h, log=config.target_log_returns
        )
        if config.target_direction:
            result = add_target_direction(
                result, horizon=h, threshold=config.target_threshold
            )
    logger.info("Target variables added for horizons %s", config.target_horizons)

    # 5. NaN handling
    rows_before = len(result)
    if config.drop_nan:
        result = result.dropna()
    rows_after = len(result)
    dropped = rows_before - rows_after
    logger.info("NaN handling: %d rows dropped (%d remaining)", dropped, rows_after)

    return result


def _validate_input(df: pd.DataFrame) -> None:
    """Raise if *df* does not meet minimum requirements."""
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns: {sorted(missing)}")
