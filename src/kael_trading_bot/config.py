"""Central configuration for the Kael Trading Bot.

Forex pairs, date ranges, intervals, and cache settings are all
driven from this single source of truth.  Values can be overridden
programmatically or via environment variables.
"""

from __future__ import annotations


import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from datetime import date
from dateutil.relativedelta import relativedelta

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_FOREX_PAIRS: list[str] = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "AUDUSD=X",
    "USDCAD=X",
    "USDCHF=X",
]

DEFAULT_START_DATE: str = "2020-01-01"
DEFAULT_END_DATE: str = "2025-01-01"
DEFAULT_INTERVAL: str = "1d"
DEFAULT_CACHE_DIR: str = ".cache/forex_data"

# Scanner defaults
DEFAULT_SCANNER_ENABLED: bool = True
DEFAULT_SCANNER_INTERVAL_MINUTES: int = 15
DEFAULT_SCANNER_DATA_DIR: str = ".cache/trade_setups"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestionConfig:
    """Immutable configuration that drives the data-ingestion pipeline.

    Attributes:
        pairs:      Yahoo Finance tickers for the forex pairs to fetch.
        start_date: ISO-8601 date string (inclusive).
        end_date:   ISO-8601 date string (inclusive).
        interval:   Pandas / yfinance frequency alias (``1d``, ``1h``, …).
        cache_dir:  Local directory used for parquet cache files.
    """

    pairs: tuple[str, ...] = tuple(DEFAULT_FOREX_PAIRS)
    start_date: str = os.getenv("KAEL_START_DATE", DEFAULT_START_DATE)
    DEFAULT_END_DATE: str = date.today()
    DEFAULT_START_DATE = DEFAULT_END_DATE - relativedelta(years=5)
    end_date: str = os.getenv("KAEL_END_DATE", DEFAULT_END_DATE)
    interval: str = os.getenv("KAEL_INTERVAL", DEFAULT_INTERVAL)
    cache_dir: str = os.getenv("KAEL_CACHE_DIR", DEFAULT_CACHE_DIR)

    # -- derived helpers ---------------------------------------------------

    @property
    def start_dt(self) -> datetime:
        """Return *start_date* as a timezone-naive ``datetime``."""
        return datetime.strptime(self.start_date, "%Y-%m-%d")

    @property
    def end_dt(self) -> datetime:
        """Return *end_date* as a timezone-naive ``datetime``."""
        return datetime.strptime(self.end_date, "%Y-%m-%d")

    @property
    def cache_path(self) -> Path:
        """Resolved ``Path`` to the cache directory."""
        return Path(self.cache_dir).expanduser().resolve()


@dataclass(frozen=True)
class ScannerConfig:
    """Configuration for the scheduled trade setup scanner.

    Attributes:
        enabled:       Whether the scanner is active.
        interval_minutes: How often (in minutes) a scan cycle runs.
        data_dir:      Directory where scanned setups are persisted.
    """

    enabled: bool = os.getenv("KAEL_SCANNER_ENABLED", str(DEFAULT_SCANNER_ENABLED)).lower() in ("1", "true", "yes")
    interval_minutes: int = int(os.getenv("KAEL_SCANNER_INTERVAL", str(DEFAULT_SCANNER_INTERVAL_MINUTES)))
    data_dir: str = os.getenv("KAEL_SCANNER_DATA_DIR", DEFAULT_SCANNER_DATA_DIR)

    @property
    def data_path(self) -> Path:
        """Resolved ``Path`` to the scanner data directory."""
        return Path(self.data_dir).expanduser().resolve()