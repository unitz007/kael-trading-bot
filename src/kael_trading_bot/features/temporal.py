"""Time-based feature extraction from OHLCV DataFrames.

Functions here extract calendar/seasonal information from the DataFrame
index (expected to be a ``DatetimeIndex``) and compute rolling-window
statistics.
"""

from __future__ import annotations

import pandas as pd


def add_time_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Append calendar-derived features.

    The following columns are added:

    * ``day_of_week`` – integer (0 = Monday, 6 = Sunday)
    * ``hour_of_day`` – integer (0–23)
    * ``minute_of_hour`` – integer (0–59)
    * ``is_month_start`` – boolean
    * ``is_month_end`` – boolean

    Parameters
    ----------
    df:
        DataFrame whose index is a ``DatetimeIndex``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with time-feature columns added.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "DataFrame index must be a DatetimeIndex to extract time features"
        )

    result = df.copy()
    idx = result.index

    result["day_of_week"] = idx.dayofweek
    result["hour_of_day"] = idx.hour
    result["minute_of_hour"] = idx.minute
    result["is_month_start"] = idx.is_month_start.astype(int)
    result["is_month_end"] = idx.is_month_end.astype(int)
    return result


def add_rolling_stats(
    df: pd.DataFrame,
    close_col: str = "close",
    windows: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """Append rolling mean, std, min, max, and skew of the close price.

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    close_col:
        Column to compute statistics over.
    windows:
        Rolling window sizes.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with columns like
        ``close_roll_<window>_mean``, ``_std``, ``_min``, ``_max``,
        ``_skew`` added.
    """
    result = df.copy()
    series = result[close_col]
    for w in windows:
        result[f"{close_col}_roll_{w}_mean"] = series.rolling(window=w, min_periods=w).mean()
        result[f"{close_col}_roll_{w}_std"] = series.rolling(window=w, min_periods=w).std()
        result[f"{close_col}_roll_{w}_min"] = series.rolling(window=w, min_periods=w).min()
        result[f"{close_col}_roll_{w}_max"] = series.rolling(window=w, min_periods=w).max()
        result[f"{close_col}_roll_{w}_skew"] = series.rolling(window=w, min_periods=w).skew()
    return result
