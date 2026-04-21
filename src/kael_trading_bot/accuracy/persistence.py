"""Append-only JSON-lines store for prediction records.

Mirrors the pattern established by
:class:`~kael_trading_bot.scanner.persistence.SetupStore` — each prediction
is stored as a single JSON object per line in ``predictions.jsonl``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from kael_trading_bot.accuracy.models import PredictionRecord

logger = logging.getLogger(__name__)


class PredictionStore:
    """Append-only JSON-lines store for prediction records.

    Parameters
    ----------
    directory:
        Directory that holds ``predictions.jsonl``.
    """

    def __init__(self, directory: str | Path = ".cache/predictions") -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "predictions.jsonl"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def save(self, prediction: PredictionRecord) -> bool:
        """Append *prediction* unless an identical record already exists.

        Idempotency key: ``pair`` + ``timeframe`` + ``predicted_at`` +
        ``horizon_at``.

        Returns ``True`` if a new record was written, ``False`` if it
        was a duplicate.
        """
        if not prediction.pair or not prediction.predicted_at or not prediction.horizon_at:
            logger.warning("Skipping prediction with missing idempotency fields.")
            return False

        # Check for duplicate
        for existing in self._read_all_records():
            if (
                existing.get("pair") == prediction.pair
                and existing.get("timeframe") == prediction.timeframe
                and existing.get("predicted_at") == prediction.predicted_at
                and existing.get("horizon_at") == prediction.horizon_at
            ):
                logger.debug(
                    "Duplicate prediction skipped: %s/%s @ %s → %s",
                    prediction.pair,
                    prediction.timeframe,
                    prediction.predicted_at,
                    prediction.horizon_at,
                )
                return False

        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(prediction.to_dict(), default=str) + "\n")

        return True

    def save_batch(self, predictions: list[PredictionRecord]) -> int:
        """Persist a batch of predictions, returning the number newly written."""
        count = 0
        for p in predictions:
            if self.save(p):
                count += 1
        return count

    def query(
        self,
        pair: Optional[str] = None,
        timeframe: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: int = 100,
    ) -> list[PredictionRecord]:
        """Return prediction records, optionally filtered.

        Results are returned in reverse-chronological order by
        ``generation_ts`` (newest first).

        Parameters
        ----------
        pair:
            Filter by forex pair (e.g. ``"EURUSD=X"``).
        timeframe:
            Filter by timeframe (e.g. ``"1h"``).
        model_name:
            Filter by model name.
        limit:
            Maximum number of records to return.
        """
        records = self._read_all_records()

        if pair is not None:
            records = [r for r in records if r.get("pair") == pair]

        if timeframe is not None:
            records = [r for r in records if r.get("timeframe") == timeframe]

        if model_name is not None:
            records = [r for r in records if r.get("model_name") == model_name]

        # Sort by generation_ts descending
        records.sort(key=lambda r: r.get("generation_ts", ""), reverse=True)

        return [PredictionRecord.from_dict(r) for r in records[:limit]]

    def get_by_id(self, prediction_id: str) -> Optional[PredictionRecord]:
        """Return a single prediction by its ID, or ``None`` if not found."""
        for rec in self._read_all_records():
            if rec.get("id") == prediction_id:
                return PredictionRecord.from_dict(rec)
        return None

    def update_actual_price(self, prediction_id: str, actual_price: float) -> bool:
        """Update the actual price for a stored prediction.

        Rewrites the entire file with the updated record. Returns ``True``
        if the record was found and updated, ``False`` otherwise.
        """
        records = self._read_all_records()
        updated = False

        for rec in records:
            if rec.get("id") == prediction_id:
                rec["actual_price"] = actual_price
                updated = True
                break

        if updated:
            self._write_all(records)

        return updated

    def clear(self) -> int:
        """Delete all stored predictions, returning the count removed."""
        records = self._read_all_records()
        count = len(records)
        if self._path.exists():
            self._path.unlink()
        return count

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _read_all_records(self) -> list[dict[str, Any]]:
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
            logger.error("Corrupt predictions file %s: %s", self._path, exc)
        return records

    def _write_all(self, records: list[dict[str, Any]]) -> None:
        """Overwrite the JSONL file with the given records."""
        with open(self._path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, default=str) + "\n")
