"""Background scheduler for periodic trade setup scanning.

Runs in a daemon thread alongside the API server.  On each cycle it
iterates over every tracked forex pair × supported timeframe, invokes
the existing ``generate_trade_setup`` logic, and persists results via
:class:`~kael_trading_bot.scanner.persistence.SetupStore`.

Errors for individual pairs/timeframes are caught and logged so that a
single failure does not abort the entire scan cycle.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from itertools import product
from typing import Any

from kael_trading_bot.config import (
    DEFAULT_FOREX_PAIRS,
    IngestionConfig,
    ScannerConfig,
)
from kael_trading_bot.features.pipeline import FeatureConfig, build_feature_matrix
from kael_trading_bot.ingestion import ForexDataFetcher
from kael_trading_bot.scanner.persistence import SetupStore
from kael_trading_bot.telegram import TelegramNotifier
from kael_trading_bot.trade_setup import generate_trade_setup
from kael_trading_bot.training.persistence import ModelPersistence

logger = logging.getLogger(__name__)

SUPPORTED_TIMEFRAMES = ("5m", "15m", "1h", "4h")


def _pair_to_model_name(pair: str, timeframe: str) -> str:
    """Derive a filesystem-safe model name from a forex pair and timeframe."""
    base = pair.replace("=", "_").replace("^", "_").lower()
    return f"{base}_{timeframe}"


class TradeSetupScanner:
    """Periodic trade setup scanner backed by a daemon thread.

    Parameters
    ----------
    scanner_config:
        :class:`~kael_trading_bot.config.ScannerConfig` instance.
    """

    def __init__(self, scanner_config: ScannerConfig | None = None) -> None:
        self._config = scanner_config or ScannerConfig()
        self._store = SetupStore(self._config.data_path)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()  # protects _running

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def store(self) -> SetupStore:
        """The underlying :class:`SetupStore` used for persistence."""
        return self._store

    def start(self) -> None:
        """Start the background scanner in a daemon thread.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                logger.info("Scanner is already running.")
                return

            if not self._config.enabled:
                logger.info("Scanner is disabled via configuration — not starting.")
                return

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="trade-setup-scanner",
                daemon=True,
            )
            self._thread.start()
            logger.info(
                "Scanner started — interval: %d minute(s), data dir: %s",
                self._config.interval_minutes,
                self._config.data_path,
            )

    def stop(self) -> None:
        """Signal the scanner thread to stop and wait for it to finish."""
        with self._lock:
            if self._thread is None:
                return
            self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=self._config.interval_minutes * 60 + 30)
            logger.info("Scanner stopped.")

    def run_once(self) -> int:
        """Execute a single scan cycle synchronously.

        Returns the number of new setups persisted.
        """
        return self._scan_cycle()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _run_loop(self) -> None:
        """Daemon-thread entry point — runs scan cycles until stopped."""
        interval_secs = self._config.interval_minutes * 60

        # Run the first scan immediately on start
        self._scan_cycle()

        while not self._stop_event.wait(timeout=interval_secs):
            self._scan_cycle()

    def _scan_cycle(self) -> int:
        """Run the full scan across all pairs × timeframes.

        Returns the number of newly persisted setups.
        """
        cycle_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        logger.info("Scan cycle %s started.", cycle_id)

        new_count = 0
        error_count = 0
        total = len(DEFAULT_FOREX_PAIRS) * len(SUPPORTED_TIMEFRAMES)

        for pair, timeframe in product(DEFAULT_FOREX_PAIRS, SUPPORTED_TIMEFRAMES):
            try:
                result = self._scan_pair_timeframe(pair, timeframe)
                if result is not None:
                    new_count += 1

            except Exception:
                error_count += 1
                logger.exception(
                    "Error scanning %s/%s — continuing with remaining pairs.",
                    pair,
                    timeframe,
                )

        logger.info(
            "Scan cycle %s completed: %d new, %d errors out of %d pair-timeframe combos.",
            cycle_id,
            new_count,
            error_count,
            total,
        )
        return new_count

    def _scan_pair_timeframe(
        self, pair: str, timeframe: str
    ) -> dict[str, Any] | None:
        """Scan a single pair+timeframe and persist the result.

        Returns the persisted setup dict, or ``None`` if no setup could
        be generated (e.g. no trained model).
        """
        model_name = _pair_to_model_name(pair, timeframe)
        persistence = ModelPersistence()

        # Check for a trained model
        versions = persistence.list_versions(model_name)
        if not versions:
            logger.debug("No model for %s/%s — skipping.", pair, timeframe)
            return None

        version = versions[-1]
        model, metadata = persistence.load(model_name, version)

        if not metadata.feature_names:
            logger.warning(
                "Model %s v%s has no feature names — skipping.", model_name, version
            )
            return None

        # Ingest & engineer features
        ingestion_cfg = IngestionConfig(pairs=(pair,))
        fetcher = ForexDataFetcher(ingestion_cfg)
        raw_df = fetcher.get(pair)
        raw_df_copy = raw_df.copy()
        raw_df.columns = [c.lower() for c in raw_df.columns]
        feature_df = build_feature_matrix(raw_df, config=FeatureConfig())

        # Generate trade setup
        setup = generate_trade_setup(
            pair=pair,
            model=model,
            metadata=metadata,
            feature_df=feature_df,
            ohlcv_df=raw_df_copy,
            model_name=model_name,
            model_version=version,
            timeframe=timeframe,
        )

        TelegramNotifier().notify_trade_setup(setup=setup)

        # Build dict with idempotency key
        detected_at = datetime.now(timezone.utc).isoformat()
        setup_dict = setup.to_dict()
        setup_dict["detected_at"] = detected_at

        # Persist
        written = self._store.save(setup_dict)
        if written:
            logger.info(
                "New setup persisted: %s/%s (conf=%.2f%%)",
                pair,
                timeframe,
                setup.confidence * 100,
            )
        return setup_dict