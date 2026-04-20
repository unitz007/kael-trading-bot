"""Persistence for scanned trade setups.

Stores trade setups as a single JSON-lines file so results survive
process restarts and can be queried by the API layer.

Each line is a JSON object with keys matching :class:`TradeSetup` fields
plus a ``detected_at`` timestamp for idempotency checks.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SetupStore:
    """Append-only JSON-lines store for trade setups.

    Parameters
    ----------
    directory:
        Directory that holds ``setups.jsonl``.
    """

    def __init__(self, directory: str | Path = ".cache/trade_setups") -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "setups.jsonl"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def save(self, setup_dict: dict[str, Any]) -> bool:
        """Append *setup_dict* unless an identical record already exists.

        Idempotency key: ``pair`` + ``timeframe`` + ``detected_at``.
        Returns ``True`` if a new record was written, ``False`` if it
        was a duplicate.
        """
        detected_at = setup_dict.get("detected_at", "")
        pair = setup_dict.get("pair", "")
        timeframe = setup_dict.get("timeframe", "")

        if not detected_at or not pair or not timeframe:
            logger.warning("Skipping setup with missing idempotency fields.")
            return False

        # Check for duplicate
        for existing in self._read_all():
            if (
                existing.get("pair") == pair
                and existing.get("timeframe") == timeframe
                and existing.get("detected_at") == detected_at
            ):
                logger.debug(
                    "Duplicate setup skipped: %s/%s @ %s",
                    pair,
                    timeframe,
                    detected_at,
                )
                return False

        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(setup_dict, default=str) + "\n")

        return True

    def save_batch(self, setups: list[dict[str, Any]]) -> int:
        """Persist a batch of setups, returning the number newly written."""
        count = 0
        for s in setups:
            if self.save(s):
                count += 1
        return count

    def query(
        self,
        pair: str | None = None,
        timeframe: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return the most recent setups, optionally filtered.

        Results are returned in reverse-chronological order (newest first).

        Parameters
        ----------
        pair:
            Filter by forex pair (e.g. ``"EURUSD=X"``).
        timeframe:
            Filter by timeframe (e.g. ``"1h"``).
        limit:
            Maximum number of records to return.
        """
        records = self._read_all()

        if pair is not None:
            records = [r for r in records if r.get("pair") == pair]

        if timeframe is not None:
            records = [r for r in records if r.get("timeframe") == timeframe]

        # Sort by detected_at descending
        records.sort(key=lambda r: r.get("detected_at", ""), reverse=True)

        return records[:limit]

    def clear(self) -> int:
        """Delete all stored setups, returning the count removed."""
        records = self._read_all()
        count = len(records)
        if self._path.exists():
            self._path.unlink()
        return count

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _read_all(self) -> list[dict[str, Any]]:
        """Read every line from the JSONL file."""
        if not self._path.exists():
            return []

        records: list[dict[str, Any]] = []
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Corrupt setups file %s: %s", self._path, exc)
        return records
