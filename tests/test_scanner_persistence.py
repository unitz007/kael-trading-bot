"""Tests for the trade setup scanner persistence layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from kael_trading_bot.scanner.persistence import SetupStore


class TestSetupStore:
    """Tests for :class:`SetupStore`."""

    def _make_store(self, tmp_path: Path) -> SetupStore:
        """Create a store backed by a temporary directory."""
        return SetupStore(directory=str(tmp_path / "setups"))

    def test_directory_created_on_init(self, tmp_path: Path) -> None:
        store_dir = tmp_path / "setups"
        SetupStore(directory=str(store_dir))
        assert store_dir.exists()
        assert store_dir.is_dir()

    def test_save_and_query(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        setup = {
            "pair": "EURUSD=X",
            "timeframe": "1h",
            "detected_at": "2025-01-15T12:00:00+00:00",
            "direction": "buy",
            "entry_price": 1.0850,
            "stop_loss": 1.0820,
            "take_profit": 1.0910,
            "confidence": 0.72,
        }
        assert store.save(setup) is True

        results = store.query()
        assert len(results) == 1
        assert results[0]["pair"] == "EURUSD=X"

    def test_idempotency(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        setup = {
            "pair": "EURUSD=X",
            "timeframe": "1h",
            "detected_at": "2025-01-15T12:00:00+00:00",
            "direction": "buy",
            "entry_price": 1.0850,
            "stop_loss": 1.0820,
            "take_profit": 1.0910,
            "confidence": 0.72,
        }
        assert store.save(setup) is True
        assert store.save(setup) is False  # duplicate

        results = store.query()
        assert len(results) == 1

    def test_different_detected_at_is_not_duplicate(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        setup_a = {
            "pair": "EURUSD=X",
            "timeframe": "1h",
            "detected_at": "2025-01-15T12:00:00+00:00",
            "direction": "buy",
        }
        setup_b = {
            "pair": "EURUSD=X",
            "timeframe": "1h",
            "detected_at": "2025-01-15T12:15:00+00:00",
            "direction": "sell",
        }
        assert store.save(setup_a) is True
        assert store.save(setup_b) is True

        results = store.query()
        assert len(results) == 2

    def test_query_filter_by_pair(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        store.save({
            "pair": "EURUSD=X",
            "timeframe": "1h",
            "detected_at": "2025-01-15T12:00:00+00:00",
            "direction": "buy",
        })
        store.save({
            "pair": "GBPUSD=X",
            "timeframe": "1h",
            "detected_at": "2025-01-15T12:00:00+00:00",
            "direction": "sell",
        })

        results = store.query(pair="EURUSD=X")
        assert len(results) == 1
        assert results[0]["pair"] == "EURUSD=X"

    def test_query_filter_by_timeframe(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        store.save({
            "pair": "EURUSD=X",
            "timeframe": "1h",
            "detected_at": "2025-01-15T12:00:00+00:00",
        })
        store.save({
            "pair": "EURUSD=X",
            "timeframe": "4h",
            "detected_at": "2025-01-15T12:00:00+00:00",
        })

        results = store.query(timeframe="4h")
        assert len(results) == 1
        assert results[0]["timeframe"] == "4h"

    def test_query_limit(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        for i in range(5):
            store.save({
                "pair": "EURUSD=X",
                "timeframe": "1h",
                "detected_at": f"2025-01-15T12:{i:02d}:00+00:00",
            })

        results = store.query(limit=3)
        assert len(results) == 3

    def test_query_returns_newest_first(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        store.save({
            "pair": "EURUSD=X",
            "timeframe": "1h",
            "detected_at": "2025-01-15T12:00:00+00:00",
        })
        store.save({
            "pair": "EURUSD=X",
            "timeframe": "1h",
            "detected_at": "2025-01-15T13:00:00+00:00",
        })

        results = store.query()
        assert results[0]["detected_at"] > results[1]["detected_at"]

    def test_save_missing_fields_returns_false(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        assert store.save({}) is False
        assert store.save({"pair": "EURUSD=X"}) is False

    def test_clear(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        store.save({
            "pair": "EURUSD=X",
            "timeframe": "1h",
            "detected_at": "2025-01-15T12:00:00+00:00",
        })
        assert store.query() != []
        removed = store.clear()
        assert removed == 1
        assert store.query() == []

    def test_save_batch(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        setups = [
            {
                "pair": f"EURUSD=X",
                "timeframe": "1h",
                "detected_at": f"2025-01-15T12:{i:02d}:00+00:00",
            }
            for i in range(3)
        ]
        count = store.save_batch(setups)
        assert count == 3

    def test_empty_store_query(self, tmp_path: Path) -> None:
        store = self._make_store(tmp_path)
        assert store.query() == []
        assert store.query(pair="EURUSD=X") == []

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        """Data written by one instance is readable by another."""
        data_dir = str(tmp_path / "shared")
        store1 = SetupStore(directory=data_dir)
        store1.save({
            "pair": "EURUSD=X",
            "timeframe": "1h",
            "detected_at": "2025-01-15T12:00:00+00:00",
            "direction": "buy",
        })

        store2 = SetupStore(directory=data_dir)
        results = store2.query()
        assert len(results) == 1
        assert results[0]["direction"] == "buy"