"""Data models for prediction accuracy tracking.

Defines the core domain objects: :class:`PredictionRecord` (a single
prediction with its measured accuracy), :class:`AccuracyStatus` (enum
for the evaluation state), :class:`AccuracySummary` (aggregated
metrics), and :class:`AccuracyTrendPoint` (time-bucketed accuracy).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AccuracyStatus(str, Enum):
    """Evaluation state of a prediction."""

    PENDING = "pending"
    CORRECT = "correct"
    INCORRECT = "incorrect"


@dataclass
class PredictionRecord:
    """A single prediction with its accuracy measurement.

    Attributes
    ----------
    id:
        Unique identifier for this prediction record.
    pair:
        Forex pair ticker (e.g. ``"EURUSD=X"``).
    timeframe:
        Timeframe of the prediction (e.g. ``"1h"``).
    predicted_direction:
        Direction predicted by the model (``"UP"``, ``"DOWN"``, ``"FLAT"``).
    predicted_at:
        Timestamp when the prediction was made.
    target_price:
        The price the model expected to reach.
    actual_price:
        The actual price at the target time (``None`` if still pending).
    percentage_drift:
        Signed percentage difference between predicted and actual price.
        ``None`` if still pending.
    status:
        Current evaluation status.
    model_name:
        Name of the model that generated the prediction.
    model_version:
        Version of the model that generated the prediction.
    created_at:
        Timestamp when this record was created.
    """

    id: str
    pair: str
    timeframe: str
    predicted_direction: str
    predicted_at: str
    target_price: float | None = None
    actual_price: float | None = None
    percentage_drift: float | None = None
    status: str = AccuracyStatus.PENDING.value
    model_name: str | None = None
    model_version: str | None = None
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary (JSON-safe)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PredictionRecord:
        """Deserialise from a plain dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AccuracySummary:
    """Aggregated accuracy metrics for a group of predictions.

    Attributes
    ----------
    pair:
        Forex pair ticker (or ``None`` for overall summary).
    timeframe:
        Timeframe (or ``None`` for overall summary).
    total_predictions:
        Total number of predictions in this group.
    evaluated_predictions:
        Number of predictions that have been evaluated (not pending).
    correct_count:
        Number of correct predictions.
    incorrect_count:
        Number of incorrect predictions.
    pending_count:
        Number of pending (unevaluated) predictions.
    win_rate:
        Fraction of evaluated predictions that were correct (0.0–1.0).
        ``None`` if no predictions have been evaluated yet.
    avg_percentage_drift:
        Average of the absolute percentage drift across evaluated predictions.
        ``None`` if no predictions have been evaluated yet.
    best_prediction:
        The prediction record with the smallest absolute drift (best).
        ``None`` if no evaluated predictions exist.
    worst_prediction:
        The prediction record with the largest absolute drift (worst).
        ``None`` if no evaluated predictions exist.
    """

    pair: str | None = None
    timeframe: str | None = None
    total_predictions: int = 0
    evaluated_predictions: int = 0
    correct_count: int = 0
    incorrect_count: int = 0
    pending_count: int = 0
    win_rate: float | None = None
    avg_percentage_drift: float | None = None
    best_prediction: dict[str, Any] | None = None
    worst_prediction: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary (JSON-safe)."""
        return asdict(self)


@dataclass
class AccuracyTrendPoint:
    """A single data-point in an accuracy trend series.

    Attributes
    ----------
    period:
        Label for the time bucket (e.g. ``"2024-W01"``, ``"2024-01-15"``).
    total_predictions:
        Number of predictions in this period.
    evaluated_predictions:
        Number of evaluated predictions in this period.
    correct_count:
        Number of correct predictions in this period.
    win_rate:
        Win rate for this period (0.0–1.0). ``None`` if none evaluated.
    avg_percentage_drift:
        Average percentage drift for this period. ``None`` if none evaluated.
    """

    period: str
    total_predictions: int = 0
    evaluated_predictions: int = 0
    correct_count: int = 0
    win_rate: float | None = None
    avg_percentage_drift: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary (JSON-safe)."""
        return asdict(self)
