"""Data models for prediction records and accuracy evaluation results.

Defines :class:`PredictionRecord` — the persisted representation of a single
model-generated prediction — and :class:`AccuracyResult` — the outcome of
comparing a prediction against actual market data.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Optional


class CorrectnessStatus(str, Enum):
    """Possible outcomes of evaluating a prediction against actual prices."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    PENDING = "pending"
    NO_DATA = "no_data"


@dataclass
class PredictionRecord:
    """A single stored prediction from the ML model.

    Attributes:
        id:                Unique identifier (UUID hex string).
        pair:              Forex pair ticker (e.g. ``"EURUSD=X"``).
        timeframe:         Timeframe of the prediction (e.g. ``"1d"``, ``"1h"``).
        direction:         Predicted price direction — ``"buy"`` or ``"sell"``.
        predicted_price:   The predicted price at the horizon.
        predicted_at:      ISO-8601 timestamp when the prediction was generated.
        horizon_at:        ISO-8601 timestamp for when the prediction targets.
        actual_price:      Actual market price at the horizon (``None`` if unknown).
        model_name:        Name of the model that generated the prediction.
        model_version:     Version string of the model.
        generation_ts:     ISO-8601 timestamp of when this record was created.
    """

    pair: str
    timeframe: str
    direction: str
    predicted_price: float
    predicted_at: str
    horizon_at: str
    model_name: str = ""
    model_version: str = ""
    id: str = ""
    actual_price: Optional[float] = None
    generation_ts: str = ""

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON persistence."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PredictionRecord:
        """Deserialise from a dict (e.g. loaded from JSON)."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AccuracyResult:
    """Result of evaluating a single prediction against actual market data.

    Attributes:
        prediction_id:        The ID of the evaluated prediction.
        pair:                 Forex pair.
        predicted_price:      The price the model predicted.
        actual_price:         The actual market price (``None`` if unavailable).
        percentage_drift:     Absolute percentage drift between predicted and actual.
        predicted_direction:  Direction the model predicted.
        actual_direction:     Direction the price actually moved (``None`` if unknown).
        directional_correct:  Whether the direction was predicted correctly.
        status:               Final correctness status.
        tolerance_pct:        The drift tolerance percentage used.
    """

    prediction_id: str
    pair: str
    predicted_price: float
    actual_price: Optional[float]
    percentage_drift: Optional[float]
    predicted_direction: str
    actual_direction: Optional[str]
    directional_correct: Optional[bool]
    status: CorrectnessStatus
    tolerance_pct: float = 2.0

    def to_dict(self) -> dict:
        """Serialise to a plain dict."""
        d = asdict(self)
        d["status"] = self.status.value
        return d
