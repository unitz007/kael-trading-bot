"""Technical indicator computations on OHLCV DataFrames.

All functions accept a pandas DataFrame with at least an 'open', 'high',
'low', 'close' (and sometimes 'volume') column and return a **new**
DataFrame with the indicator columns appended.  The original data is never
mutated.
"""

from __future__ import annotations

import pandas as pd


def add_sma(
    df: pd.DataFrame,
    close_col: str = "close",
    periods: tuple[int, ...] = (10, 20, 50),
) -> pd.DataFrame:
    """Append Simple Moving Average columns.

    Parameters
    ----------
    df:
        OHLCV DataFrame with a ``close_col`` column.
    close_col:
        Name of the close-price column.
    periods:
        Window sizes to compute.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``sma_<period>`` columns added.
    """
    result = df.copy()
    for p in periods:
        result[f"sma_{p}"] = result[close_col].rolling(window=p, min_periods=p).mean()
    return result


def add_ema(
    df: pd.DataFrame,
    close_col: str = "close",
    periods: tuple[int, ...] = (9, 21),
) -> pd.DataFrame:
    """Append Exponential Moving Average columns.

    Parameters
    ----------
    df:
        OHLCV DataFrame with a ``close_col`` column.
    close_col:
        Name of the close-price column.
    periods:
        Span sizes to compute.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``ema_<period>`` columns added.
    """
    result = df.copy()
    for p in periods:
        result[f"ema_{p}"] = result[close_col].ewm(span=p, adjust=False).mean()
    return result


def add_rsi(
    df: pd.DataFrame,
    close_col: str = "close",
    period: int = 14,
) -> pd.DataFrame:
    """Append a Relative Strength Index column.

    Uses the standard Wilder smoothing method (equivalent to an EMA with
    ``com = period - 1``).

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    close_col:
        Name of the close-price column.
    period:
        Look-back window.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an ``rsi_<period>`` column added.
    """
    result = df.copy()
    delta = result[close_col].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("inf"))
    result[f"rsi_{period}"] = 100.0 - (100.0 / (1.0 + rs))
    return result


def add_macd(
    df: pd.DataFrame,
    close_col: str = "close",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    prefix: str = "macd",
) -> pd.DataFrame:
    """Append MACD, signal line, and histogram columns.

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    close_col:
        Name of the close-price column.
    fast, slow, signal:
        EMA spans for the MACD calculation.
    prefix:
        Column-name prefix (default ``macd``).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``<prefix>_<suffix>`` columns added:
        ``_line``, ``_signal``, ``_hist``.
    """
    result = df.copy()
    fast_ema = result[close_col].ewm(span=fast, adjust=False).mean()
    slow_ema = result[close_col].ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    result[f"{prefix}_line"] = macd_line
    result[f"{prefix}_signal"] = signal_line
    result[f"{prefix}_hist"] = macd_line - signal_line
    return result


def add_bollinger_bands(
    df: pd.DataFrame,
    close_col: str = "close",
    period: int = 20,
    num_std: float = 2.0,
    prefix: str = "bb",
) -> pd.DataFrame:
    """Append Bollinger Bands (upper, middle, lower, bandwidth, %B).

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    close_col:
        Name of the close-price column.
    period:
        Rolling window for the middle band (SMA).
    num_std:
        Number of standard deviations for the bands.
    prefix:
        Column-name prefix (default ``bb``).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``<prefix>_upper``, ``<prefix>_middle``,
        ``<prefix>_lower``, ``<prefix>_bw``, ``<prefix>_pct_b`` columns.
    """
    result = df.copy()
    middle = result[close_col].rolling(window=period, min_periods=period).mean()
    std = result[close_col].rolling(window=period, min_periods=period).std()

    upper = middle + num_std * std
    lower = middle - num_std * std
    bw = upper - lower  # bandwidth
    pct_b = (result[close_col] - lower) / (bw.replace(0, float("nan")))

    result[f"{prefix}_upper"] = upper
    result[f"{prefix}_middle"] = middle
    result[f"{prefix}_lower"] = lower
    result[f"{prefix}_bw"] = bw
    result[f"{prefix}_pct_b"] = pct_b
    return result


def add_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """Append an Average True Range column.

    Parameters
    ----------
    df:
        OHLCV DataFrame with ``high``, ``low``, and ``close`` columns.
    period:
        Look-back window.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an ``atr_<period>`` column added.
    """
    result = df.copy()
    high = result["high"]
    low = result["low"]
    prev_close = result["close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    result[f"atr_{period}"] = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return result
