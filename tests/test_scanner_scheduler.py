"""Tests for the trade setup scanner scheduler."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kael_trading_bot.scanner.scheduler import TradeSetupScanner
from kael_trading_bot.scanner.persistence import SetupStore


class TestTradeSetupScanner:
    """Tests for :class:`TradeSetupScanner`."""

    def test_start_when_disabled_does_not_run(self, tmp_path: Path) -> None:
        from kael_trading_bot.config import ScannerConfig

        cfg = ScannerConfig(
            enabled=False,
            interval_minutes=1,
            data_dir=str(tmp_path / "disabled"),
        )
        scanner = TradeSetupScanner(cfg)
        scanner.start()

        # Give it a moment — no thread should exist
        time.sleep(0.2)
        assert scanner._thread is None

    def test_start_creates_daemon_thread(self, tmp_path: Path) -> None:
        from kael_trading_bot.config import ScannerConfig

        cfg = ScannerConfig(
            enabled=True,
            interval_minutes=60,  # long interval to prevent extra runs
            data_dir=str(tmp_path / "scanner"),
        )

        # Patch _scan_cycle so it returns immediately without real work
        with patch.object(
            TradeSetupScanner, "_scan_cycle", return_value=0
        ):
            scanner = TradeSetupScanner(cfg)
            scanner.start()
            time.sleep(0.3)

            assert scanner._thread is not None
            assert scanner._thread.daemon is True
            scanner.stop()

    def test_stop_signals_thread(self, tmp_path: Path) -> None:
        from kael_trading_bot.config import ScannerConfig

        cfg = ScannerConfig(
            enabled=True,
            interval_minutes=60,
            data_dir=str(tmp_path / "scanner"),
        )

        with patch.object(
            TradeSetupScanner, "_scan_cycle", return_value=0
        ):
            scanner = TradeSetupScanner(cfg)
            scanner.start()
            time.sleep(0.2)

            scanner.stop()
            # After stop, the thread should have finished
            assert not scanner._thread.is_alive()

    def test_store_property_returns_setup_store(self, tmp_path: Path) -> None:
        from kael_trading_bot.config import ScannerConfig

        cfg = ScannerConfig(data_dir=str(tmp_path / "store_test"))
        scanner = TradeSetupScanner(cfg)
        assert isinstance(scanner.store, SetupStore)

    def test_scan_cycle_logs_and_counts(self, tmp_path: Path) -> None:
        from kael_trading_bot.config import ScannerConfig

        cfg = ScannerConfig(
            enabled=True,
            interval_minutes=60,
            data_dir=str(tmp_path / "cycle_test"),
        )
        scanner = TradeSetupScanner(cfg)

        # Patch _scan_pair_timeframe to return a result for one combo
        # and None for the rest (no model available)
        def fake_scan(pair: str, timeframe: str):
            if pair == "EURUSD=X" and timeframe == "1h":
                return {"pair": pair, "timeframe": timeframe}
            return None

        with patch.object(scanner, "_scan_pair_timeframe", side_effect=fake_scan):
            count = scanner._scan_cycle()

        # Should have persisted the one result
        assert count == 1
        results = scanner.store.query()
        assert len(results) == 1

    def test_scan_cycle_continues_on_error(self, tmp_path: Path) -> None:
        from kael_trading_bot.config import ScannerConfig

        cfg = ScannerConfig(
            enabled=True,
            interval_minutes=60,
            data_dir=str(tmp_path / "error_test"),
        )
        scanner = TradeSetupScanner(cfg)

        call_count = 0

        def flaky_scan(pair: str, timeframe: str):
            nonlocal call_count
            call_count += 1
            if pair == "GBPUSD=X":
                raise RuntimeError("Network error")
            return {"pair": pair, "timeframe": timeframe}

        with patch.object(scanner, "_scan_pair_timeframe", side_effect=flaky_scan):
            count = scanner._scan_cycle()

        # Should still succeed for non-erroring pairs
        assert count > 0
        # GBPUSD=X should have been attempted (6 pairs × 4 timeframes = 24 combos)
        # Plus it should have continued past the error
        assert call_count == 24  # all combos attempted

    def test_run_once(self, tmp_path: Path) -> None:
        from kael_trading_bot.config import ScannerConfig

        cfg = ScannerConfig(data_dir=str(tmp_path / "once"))
        scanner = TradeSetupScanner(cfg)

        with patch.object(scanner, "_scan_cycle", return_value=3):
            result = scanner.run_once()

        assert result == 3
