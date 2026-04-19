"""Tests for the forex data ingestion module.

These tests avoid hitting the network by mocking ``yfinance`` responses
and exercising the full validation / caching pipeline.
"""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from kael_trading_bot.config import DEFAULT_FOREX_PAIRS, IngestionConfig
from kael_trading_bot.ingestion import (
    OHLCV_COLUMNS,
    ForexDataFetcher,
    _cache_path_for_pair,
    _coerce_dtypes,
    _ensure_chronological,
    _ensure_columns,
    _handle_missing,
    _validate_ohlcv_logic,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ohlcv(
    index: pd.DatetimeIndex | None = None,
    nrows: int = 10,
    pair: str = "EURUSD=X",
) -> pd.DataFrame:
    """Create a well-formed OHLCV DataFrame for testing."""
    if index is None:
        index = pd.date_range("2024-01-01", periods=nrows, freq="D")
    data = {
        "Open": np.random.default_rng(42).uniform(1.0, 2.0, nrows),
        "High": np.random.default_rng(43).uniform(2.0, 3.0, nrows),
        "Low": np.random.default_rng(44).uniform(0.5, 1.0, nrows),
        "Close": np.random.default_rng(45).uniform(1.0, 2.0, nrows),
        "Volume": np.random.default_rng(46).integers(1000, 100_000, nrows).astype(float),
    }
    df = pd.DataFrame(data, index=index)
    # Ensure High >= Low, High >= Open/Close, Low <= Open/Close
    df["High"] = df[["Open", "Close", "High"]].max(axis=1)
    df["Low"] = df[["Open", "Close", "Low"]].min(axis=1)
    return df


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return _make_ohlcv()


@pytest.fixture
def tmp_cache(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture
def config(tmp_cache: Path) -> IngestionConfig:
    return IngestionConfig(
        pairs=("EURUSD=X", "GBPUSD=X"),
        start_date="2024-01-01",
        end_date="2024-12-31",
        interval="1d",
        cache_dir=str(tmp_cache),
    )


# ---------------------------------------------------------------------------
# Unit tests: validation helpers
# ---------------------------------------------------------------------------


class TestEnsureColumns:
    def test_keeps_only_ohlcv(self, sample_df: pd.DataFrame) -> None:
        sample_df["Extra"] = 99.0
        result = _ensure_columns(sample_df, "EURUSD=X")
        assert list(result.columns) == OHLCV_COLUMNS
        assert "Extra" not in result.columns

    def test_raises_on_missing_column(self) -> None:
        df = pd.DataFrame({"Open": [1.0], "High": [2.0]})
        with pytest.raises(ValueError, match="Missing OHLCV columns"):
            _ensure_columns(df, "X")


class TestCoerceDtypes:
    def test_correct_types(self, sample_df: pd.DataFrame) -> None:
        result = _coerce_dtypes(sample_df, "EURUSD=X")
        for col in OHLCV_COLUMNS:
            assert result[col].dtype == np.float64

    def test_raises_on_bad_cast(self) -> None:
        df = pd.DataFrame({"Open": ["not_a_number"]}, index=pd.date_range("2024-01-01", periods=1))
        for col in OHLCV_COLUMNS:
            if col != "Open":
                df[col] = 1.0
        with pytest.raises(TypeError, match="Cannot cast"):
            _coerce_dtypes(df, "X")


class TestHandleMissing:
    def test_ffill(self, sample_df: pd.DataFrame) -> None:
        sample_df.loc[sample_df.index[2], "Close"] = np.nan
        result = _handle_missing(sample_df, strategy="ffill", pair="X")
        assert result.isna().sum().sum() == 0

    def test_drop(self, sample_df: pd.DataFrame) -> None:
        sample_df.loc[sample_df.index[2], "Close"] = np.nan
        result = _handle_missing(sample_df, strategy="drop", pair="X")
        assert len(result) == len(sample_df) - 1
        assert result.isna().sum().sum() == 0

    def test_raise(self, sample_df: pd.DataFrame) -> None:
        sample_df.loc[sample_df.index[2], "Close"] = np.nan
        with pytest.raises(ValueError, match="missing values"):
            _handle_missing(sample_df, strategy="raise", pair="X")

    def test_no_missing_is_noop(self, sample_df: pd.DataFrame) -> None:
        result = _handle_missing(sample_df, strategy="ffill", pair="X")
        pd.testing.assert_frame_equal(result, sample_df)


class TestEnsureChronological:
    def test_sorts_reverse(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="D")[::-1]
        df = _make_ohlcv(index=idx)
        result = _ensure_chronological(df, "X")
        assert result.index.is_monotonic_increasing

    def test_already_sorted(self, sample_df: pd.DataFrame) -> None:
        result = _ensure_chronological(sample_df, "X")
        assert result.index.is_monotonic_increasing

    def test_converts_non_datetime_index(self) -> None:
        df = _make_ohlcv()
        df.index = df.index.strftime("%Y-%m-%d")
        result = _ensure_chronological(df, "X")
        assert isinstance(result.index, pd.DatetimeIndex)


class TestValidateOhlcvLogic:
    def test_valid_data_passes(self, sample_df: pd.DataFrame) -> None:
        _validate_ohlcv_logic(sample_df, "X")

    def test_high_less_than_low_raises(self, sample_df: pd.DataFrame) -> None:
        sample_df.loc[sample_df.index[0], "High"] = 0.5
        sample_df.loc[sample_df.index[0], "Low"] = 2.0
        with pytest.raises(ValueError, match="High < Low"):
            _validate_ohlcv_logic(sample_df, "X")

    def test_high_less_than_open_raises(self, sample_df: pd.DataFrame) -> None:
        row = sample_df.index[0]
        sample_df.loc[row, "Open"] = 5.0
        sample_df.loc[row, "High"] = 3.0
        sample_df.loc[row, "Low"] = 1.0
        sample_df.loc[row, "Close"] = 2.0
        with pytest.raises(ValueError, match="High < Open"):
            _validate_ohlcv_logic(sample_df, "X")

    def test_low_greater_than_close_raises(self, sample_df: pd.DataFrame) -> None:
        row = sample_df.index[0]
        sample_df.loc[row, "High"] = 5.0
        sample_df.loc[row, "Low"] = 4.0
        sample_df.loc[row, "Close"] = 2.0
        sample_df.loc[row, "Open"] = 1.0
        with pytest.raises(ValueError, match="Low > Close"):
            _validate_ohlcv_logic(sample_df, "X")


# ---------------------------------------------------------------------------
# Unit tests: caching
# ---------------------------------------------------------------------------


class TestCachePathForPair:
    def test_deterministic_path(self, tmp_path: Path) -> None:
        p1 = _cache_path_for_pair(tmp_path, "EURUSD=X", "1d", "2024-01-01", "2024-12-31")
        p2 = _cache_path_for_pair(tmp_path, "EURUSD=X", "1d", "2024-01-01", "2024-12-31")
        assert p1 == p2

    def test_different_pair_different_path(self, tmp_path: Path) -> None:
        p1 = _cache_path_for_pair(tmp_path, "EURUSD=X", "1d", "2024-01-01", "2024-12-31")
        p2 = _cache_path_for_pair(tmp_path, "GBPUSD=X", "1d", "2024-01-01", "2024-12-31")
        assert p1 != p2

    def test_has_parquet_extension(self, tmp_path: Path) -> None:
        p = _cache_path_for_pair(tmp_path, "EURUSD=X", "1d", "2024-01-01", "2024-12-31")
        assert p.suffix == ".parquet"


# ---------------------------------------------------------------------------
# Integration tests: ForexDataFetcher (mocked yfinance)
# ---------------------------------------------------------------------------


class TestForexDataFetcher:
    def _mock_yf_ticker(self, df: pd.DataFrame) -> MagicMock:
        """Return a mocked yf.Ticker whose .history() returns *df*."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df
        return mock_ticker

    def test_get_fetches_and_caches(
        self, config: IngestionConfig, sample_df: pd.DataFrame, tmp_cache: Path
    ) -> None:
        fetcher = ForexDataFetcher(config)

        with patch("kael_trading_bot.ingestion.yf.Ticker", return_value=self._mock_yf_ticker(sample_df)):
            result = fetcher.get("EURUSD=X")

        assert not result.empty
        assert list(result.columns) == OHLCV_COLUMNS
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.is_monotonic_increasing

        # Verify a parquet file was written
        parquet_files = list(tmp_cache.glob("*.parquet"))
        assert len(parquet_files) == 1

    def test_get_loads_from_cache(
        self, config: IngestionConfig, sample_df: pd.DataFrame, tmp_cache: Path
    ) -> None:
        fetcher = ForexDataFetcher(config)

        with patch("kael_trading_bot.ingestion.yf.Ticker", return_value=self._mock_yf_ticker(sample_df)):
            first = fetcher.get("EURUSD=X")

        # Second call should load from cache without calling yfinance
        with patch("kael_trading_bot.ingestion.yf.Ticker") as mock_ticker_cls:
            second = fetcher.get("EURUSD=X")
            mock_ticker_cls.assert_not_called()

        pd.testing.assert_frame_equal(first, second)

    def test_get_raises_on_empty_response(self, config: IngestionConfig) -> None:
        fetcher = ForexDataFetcher(config)
        empty_df = pd.DataFrame()

        with patch(
            "kael_trading_bot.ingestion.yf.Ticker",
            return_value=self._mock_yf_ticker(empty_df),
        ):
            with pytest.raises(ValueError, match="No data returned"):
                fetcher.get("EURUSD=X")

    def test_get_all_returns_dict(
        self, config: IngestionConfig, sample_df: pd.DataFrame
    ) -> None:
        fetcher = ForexDataFetcher(config)

        with patch(
            "kael_trading_bot.ingestion.yf.Ticker",
            return_value=self._mock_yf_ticker(sample_df),
        ):
            results = fetcher.get_all()

        assert set(results.keys()) == set(config.pairs)
        for df in results.values():
            assert list(df.columns) == OHLCV_COLUMNS

    def test_get_all_skips_failed_pairs(
        self, config: IngestionConfig, sample_df: pd.DataFrame
    ) -> None:
        fetcher = ForexDataFetcher(config)

        call_count = 0

        def _ticker_side_effect(*args, **kwargs):  # noqa: ANN002, ANN003
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Network error")
            return self._mock_yf_ticker(sample_df)

        with patch(
            "kael_trading_bot.ingestion.yf.Ticker",
            side_effect=_ticker_side_effect,
        ):
            results = fetcher.get_all()

        # One pair failed, one succeeded
        assert len(results) == 1

    def test_invalidate_cache_pair(
        self, config: IngestionConfig, sample_df: pd.DataFrame, tmp_cache: Path
    ) -> None:
        fetcher = ForexDataFetcher(config)

        with patch("kael_trading_bot.ingestion.yf.Ticker", return_value=self._mock_yf_ticker(sample_df)):
            fetcher.get("EURUSD=X")

        assert list(tmp_cache.glob("*.parquet"))

        fetcher.invalidate_cache("EURUSD=X")
        assert not list(tmp_cache.glob("*.parquet"))

    def test_invalidate_cache_all(
        self, config: IngestionConfig, sample_df: pd.DataFrame, tmp_cache: Path
    ) -> None:
        fetcher = ForexDataFetcher(config)

        with patch("kael_trading_bot.ingestion.yf.Ticker", return_value=self._mock_yf_ticker(sample_df)):
            fetcher.get("EURUSD=X")
            fetcher.get("GBPUSD=X")

        assert len(list(tmp_cache.glob("*.parquet"))) == 2

        fetcher.invalidate_cache()
        assert not list(tmp_cache.glob("*.parquet"))

    def test_unconfigured_pair_warns_but_still_fetches(
        self, config: IngestionConfig, sample_df: pd.DataFrame
    ) -> None:
        fetcher = ForexDataFetcher(config)
        with patch(
            "kael_trading_bot.ingestion.yf.Ticker",
            return_value=self._mock_yf_ticker(sample_df),
        ):
            result = fetcher.get("USDJPY=X")  # not in config
        assert not result.empty


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestIngestionConfig:
    def test_defaults(self) -> None:
        cfg = IngestionConfig()
        assert cfg.pairs == tuple(DEFAULT_FOREX_PAIRS)
        assert cfg.start_date == "2020-01-01"
        assert cfg.end_date == "2025-01-01"
        assert cfg.interval == "1d"

    def test_custom_pairs(self) -> None:
        cfg = IngestionConfig(pairs=("EURUSD=X",))
        assert cfg.pairs == ("EURUSD=X",)

    def test_start_dt(self) -> None:
        cfg = IngestionConfig(start_date="2024-06-15")
        assert cfg.start_dt == datetime(2024, 6, 15)

    def test_end_dt(self) -> None:
        cfg = IngestionConfig(end_date="2024-12-31")
        assert cfg.end_dt == datetime(2024, 12, 31)

    def test_cache_path(self) -> None:
        cfg = IngestionConfig(cache_dir="/tmp/forex")
        assert cfg.cache_path == Path("/tmp/forex")

    def test_frozen(self) -> None:
        cfg = IngestionConfig()
        with pytest.raises(AttributeError):
            cfg.pairs = ("X",)  # type: ignore[misc]
