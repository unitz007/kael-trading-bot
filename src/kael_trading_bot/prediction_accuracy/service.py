"""Prediction accuracy service interface.

Provides :class:`PredictionAccuracyService` — the service that the API
layer depends on for querying prediction accuracy data.  The concrete
implementation lives in the prediction persistence / accuracy-calculation
module (task #94).  Here we expose a clear interface so the API can be
built and tested independently.

The service is instantiated lazily via :func:`get_accuracy_service` so
that tests can inject a mock without touching module-level state.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from kael_trading_bot.prediction_accuracy.models import (
    AccuracySummary,
    AccuracyTrendPoint,
    PredictionRecord,
)

logger = logging.getLogger(__name__)


class PredictionAccuracyService(ABC):
    """Abstract base class for prediction accuracy operations.

    Subclass this to provide a concrete implementation backed by the
    persistence layer from task #94.
    """

    # ------------------------------------------------------------------ #
    # Query methods consumed by the API layer
    # ------------------------------------------------------------------ #

    @abstractmethod
    def list_predictions(
        self,
        *,
        pair: str | None = None,
        timeframe: str | None = None,
        status: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """Return a paginated list of prediction records.

        Parameters
        ----------
        pair:
            Filter by forex pair (e.g. ``"EURUSD=X"``).
        timeframe:
            Filter by timeframe (e.g. ``"1h"``).
        status:
            Filter by correctness status — ``"correct"``, ``"incorrect"``,
            or ``"pending"``.
        page:
            1-based page number.
        page_size:
            Number of records per page.

        Returns
        -------
        dict
            ``{"predictions": [...], "total": int, "page": int, "page_size": int, "total_pages": int}``
        """

    @abstractmethod
    def get_summary(
        self,
        *,
        pair: str | None = None,
        timeframe: str | None = None,
    ) -> AccuracySummary:
        """Return aggregated accuracy metrics.

        Parameters
        ----------
        pair:
            Filter by forex pair. ``None`` for overall summary.
        timeframe:
            Filter by timeframe. ``None`` for overall summary.

        Returns
        -------
        AccuracySummary
            Aggregated metrics including win rate, avg drift, best/worst.
        """

    @abstractmethod
    def get_trend(
        self,
        *,
        pair: str | None = None,
        timeframe: str | None = None,
        period: str = "week",
    ) -> list[AccuracyTrendPoint]:
        """Return accuracy trend data over time.

        Parameters
        ----------
        pair:
            Filter by forex pair. ``None`` for all pairs.
        timeframe:
            Filter by timeframe. ``None`` for all timeframes.
        period:
            Grouping granularity — ``"day"`` or ``"week"``.

        Returns
        -------
        list[AccuracyTrendPoint]
            One entry per time bucket, in chronological order.
        """


# ------------------------------------------------------------------ #
# Module-level singleton (lazy, replaceable for testing)
# ------------------------------------------------------------------ #

_service: PredictionAccuracyService | None = None


def get_accuracy_service() -> PredictionAccuracyService:
    """Return the module-level accuracy service instance.

    If the service has not been set (e.g. in production before task #94
    is complete), a :class:`StubPredictionAccuracyService` is returned
    so the API endpoints always have something to call without raising
    ``NoneType`` errors.

    Use :func:`set_accuracy_service` to inject a real implementation
    or a mock in tests.
    """
    global _service
    if _service is None:
        _service = StubPredictionAccuracyService()
    return _service


def set_accuracy_service(svc: PredictionAccuracyService) -> None:
    """Replace the module-level accuracy service (for testing / DI)."""
    global _service
    _service = svc


class StubPredictionAccuracyService(PredictionAccuracyService):
    """Minimal no-op implementation used before task #94 lands.

    Returns empty results so the API endpoints can function and be
    tested without a real persistence back-end.
    """

    def list_predictions(
        self,
        *,
        pair: str | None = None,
        timeframe: str | None = None,
        status: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        return {
            "predictions": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
            "total_pages": 0,
        }

    def get_summary(
        self,
        *,
        pair: str | None = None,
        timeframe: str | None = None,
    ) -> AccuracySummary:
        return AccuracySummary(pair=pair, timeframe=timeframe)

    def get_trend(
        self,
        *,
        pair: str | None = None,
        timeframe: str | None = None,
        period: str = "week",
    ) -> list[AccuracyTrendPoint]:
        return []
