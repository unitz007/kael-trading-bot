"""Forex OHLCV data ingestion module.

Provides ``ForexDataFetcher`` — the single entry-point for downloading,
validating, caching, and serving historical OHLCV data for configured
forex currency pairs.

Typical usage
-------------
>>> from kael_trading_bot.config import IngestionConfig
>>> from kael_trading_bot.ingestion import ForexDataFetcher
>>> cfg = IngestionConfig(pairs=("EURUSD=X",), start_date="2023-01-01")
>>> fetcher = ForexDataFetcher(cfg)
>>> df = fetcher.get("EURUSD=X")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import yfinance as yf

from kael_trading_bot.config import IngestionConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column / type contract
# ---------------------------------------------------------------------------

OHLCV_COLUMNS: list[str] = ["Open", "High", "Low", "Close", "Volume"]
OHLCV_DTYPES: dict[str, str] = {
    "Open": "float64",
    "High": "float64",
    "Low": "float64",
    "Close": "float64",
    "Volume": "float64",
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _ensure_columns(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Keep only the five OHLCV columns; raise if any are missing."""
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing OHLCV columns {missing} for pair {pair!r}. "
            f"Available columns: {list(df.columns)}"
        )
    return df[OHLCV_COLUMNS].copy()


def _coerce_dtypes(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Cast every OHLCV column to its expected dtype."""
    for col, expected in OHLCV_DTYPES.items():
        try:
            df[col] = df[col].astype(expected)
        except (ValueError, TypeError) as exc:
            raise TypeError(
                f"Cannot cast column {col!r} to {expected} for pair {pair!r}: {exc}"
            ) from exc
    return df


def _handle_missing(
    df: pd.DataFrame,
    strategy: Literal["ffill", "drop", "raise"] = "ffill",
    pair: str = "",
) -> pd.DataFrame:
    """Deal with NaN values according to *strategy*.

    * ``ffill`` — forward-fill then back-fill any leading NaNs.
    * ``drop``  — drop rows containing NaNs.
    * ``raise`` — raise a ``ValueError``.
    """
    n_missing = int(df.isna().any(axis=1).sum())
    if n_missing == 0:
        return df

    logger.info(
        "Pair %s: %d rows with missing values (strategy=%s)",
        pair,
        n_missing,
        strategy,
    )

    if strategy == "ffill":
        df = df.ffill().bfill()
        return df
    if strategy == "drop":
        df = df.dropna()
        return df
    raise ValueError(
        f"Found {n_missing} rows with missing values for pair {pair!r} "
        f"and strategy is 'raise'."
    )


def _ensure_chronological(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Sort by index (DatetimeIndex) and verify monotonic increase."""
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as exc:
            raise TypeError(
                f"Cannot convert index to DatetimeIndex for pair {pair!r}: {exc}"
            ) from exc

    df = df.sort_index()
    if not df.index.is_monotonic_increasing:
        logger.warning(
            "Pair %s: index was not in chronological order — sorted.", pair
        )
    return df


def _validate_ohlcv_logic(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Basic sanity checks: High >= Low, High >= Open/Close, Low <= Open/Close."""
    violations: list[str] = []

    if (df["High"] < df["Low"]).any():
        violations.append("High < Low")

    if (df["High"] < df["Open"]).any():
        violations.append("High < Open")

    if (df["High"] < df["Close"]).any():
        violations.append("High < Close")

    if (df["Low"] > df["Open"]).any():
        violations.append("Low > Open")

    if (df["Low"] > df["Close"]).any():
        violations.append("Low > Close")

    if violations:
        raise ValueError(
            f"OHLCV logic violations for pair {pair!r}: {', '.join(violations)}"
        )

    return df


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def _cache_path_for_pair(
    cache_dir: Path, pair: str, interval: str, start: str, end: str
) -> Path:
    """Return a deterministic parquet file path for a given query."""
    safe_pair = pair.replace("=", "_").replace("^", "_")
    return cache_dir / f"{safe_pair}_{interval}_{start}_{end}.parquet"


# ---------------------------------------------------------------------------
# Main fetcher
# ---------------------------------------------------------------------------


class ForexDataFetcher:
    """Download, validate, cache, and serve OHLCV data.

    Parameters
    ----------
    config:
        An :class:`~kael_trading_bot.config.IngestionConfig` instance.
    missing_strategy:
        How to handle NaN values — ``ffill`` (default), ``drop``, or ``raise``.
    """

    def __init__(
        self,
        config: IngestionConfig,
        missing_strategy: Literal["ffill", "drop", "raise"] = "ffill",
    ) -> None:
        self._config = config
        self._missing_strategy = missing_strategy
        self._cache_dir: Path = config.cache_path
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # -- public API --------------------------------------------------------

    def get(self, pair: str) -> pd.DataFrame:
        """Return a validated ``DataFrame`` of OHLCV data for *pair*.

        If a cached parquet file exists and covers the requested date range
        it is loaded directly; otherwise the data is fetched from Yahoo
        Finance, validated, cached, and returned.
        """
        if pair not in self._config.pairs:
            logger.warning(
                "Pair %r is not in the configured pairs list. Fetching anyway.",
                pair,
            )

        cache_file = _cache_path_for_pair(
            self._cache_dir,
            pair,
            self._config.interval,
            self._config.start_date,
            self._config.end_date,
        )

        if cache_file.exists():
            logger.info("Loading %s from cache: %s", pair, cache_file)
            df = pd.read_parquet(cache_file)
        else:
            df = self._fetch_from_source(pair)
            df.to_parquet(cache_file, index=True)
            logger.info("Cached %s → %s", pair, cache_file)

        return df

    def get_all(self) -> dict[str, pd.DataFrame]:
        """Fetch (or load from cache) OHLCV data for **every** configured pair.

        Returns a dict mapping pair ticker → DataFrame.
        """
        results: dict[str, pd.DataFrame] = {}
        for pair in self._config.pairs:
            try:
                results[pair] = self.get(pair)
            except Exception:
                logger.exception("Failed to fetch pair %s — skipping.", pair)
        return results

    def invalidate_cache(self, pair: Optional[str] = None) -> None:
        """Remove cached files.

        If *pair* is ``None`` the **entire** cache directory is wiped.
        """
        if pair is None:
            for f in self._cache_dir.glob("*.parquet"):
                f.unlink(missing_ok=True)
            logger.info("Cache cleared: %s", self._cache_dir)
        else:
            cache_file = _cache_path_for_pair(
                self._cache_dir,
                pair,
                self._config.interval,
                self._config.start_date,
                self._config.end_date,
            )
            cache_file.unlink(missing_ok=True)
            logger.info("Cache invalidated for %s: %s", pair, cache_file)

    # -- private -----------------------------------------------------------

    def _fetch_from_source(self, pair: str) -> pd.DataFrame:
        """Download data via ``yfinance`` and run the full validation chain."""
        logger.info(
            "Fetching %s | %s → %s | interval=%s",
            pair,
            self._config.start_date,
            self._config.end_date,
            self._config.interval,
        )

        ticker = yf.Ticker(pair)
        df: pd.DataFrame = ticker.history(
            start=self._config.start_date,
            end=self._config.end_date,
            interval=self._config.interval,
            auto_adjust=False,
        )

        if df.empty:
            raise ValueError(
                f"No data returned for pair {pair!r} between "
                f"{self._config.start_date} and {self._config.end_date} "
                f"(interval={self._config.interval!r})."
            )

        df.index.name = "Date"
        df = df.rename(columns=str)  # ensure plain str column names

        # Validation pipeline
        df = _ensure_columns(df, pair)
        df = _coerce_dtypes(df, pair)
        df = _handle_missing(df, strategy=self._missing_strategy, pair=pair)
        df = _ensure_chronological(df, pair)
        df = _validate_ohlcv_logic(df, pair)

        logger.info(
            "Fetched & validated %s: %d rows, %s → %s",
            pair,
            len(df),
            df.index.min().date(),
            df.index.max().date(),
        )
        return df
