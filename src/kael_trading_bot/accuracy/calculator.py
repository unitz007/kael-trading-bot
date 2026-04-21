"""Accuracy calculation service for predictions.

Compares stored predictions against actual market prices and computes
percentage drift, directional correctness, and an overall correctness status.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from kael_trading_bot.accuracy.models import (
    AccuracyResult,
    CorrectnessStatus,
    PredictionRecord,
)

logger = logging.getLogger(__name__)


def calculate_percentage_drift(
    predicted_price: float, actual_price: float
) -> float:
    """Calculate percentage drift between predicted and actual prices.

    Formula: ``|predicted_price - actual_price| / actual_price × 100``

    Parameters
    ----------
    predicted_price:
        The price the model predicted.
    actual_price:
        The actual market price.

    Returns
    -------
    float
        Absolute percentage drift.

    Raises
    ------
    ValueError
        If *actual_price* is zero or negative.
    """
    if actual_price <= 0:
        raise ValueError(
            f"actual_price must be positive, got {actual_price}"
        )
    return abs(predicted_price - actual_price) / actual_price * 100


def determine_direction(
    price_at_generation: float, price_at_horizon: float
) -> str:
    """Determine the direction of price movement between two timestamps.

    Returns ``"buy"`` if the price went up, ``"sell"`` if it went down,
    or ``"flat"`` if prices are identical.
    """
    if price_at_horizon > price_at_generation:
        return "buy"
    elif price_at_horizon < price_at_generation:
        return "sell"
    return "flat"


def check_directional_correctness(
    predicted_direction: str, actual_direction: str
) -> bool:
    """Check whether the predicted direction matches the actual direction.

    A ``"flat"`` actual direction is considered **incorrect** for any
    non-flat prediction.  A ``"flat"`` prediction is considered incorrect
    for any non-flat actual direction.
    """
    if actual_direction == "flat":
        return predicted_direction == "flat"
    if predicted_direction == "flat":
        return False
    return predicted_direction == actual_direction


def get_current_status(horizon_at: str) -> str:
    """Determine whether a prediction's horizon is still in the future.

    Returns ``"pending"`` if the horizon timestamp is in the future,
    ``"expired"`` if it has passed.

    Parameters
    ----------
    horizon_at:
        ISO-8601 timestamp string.
    """
    try:
        horizon_dt = datetime.fromisoformat(horizon_at)
        if horizon_dt.tzinfo is None:
            horizon_dt = horizon_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return "pending" if horizon_dt > now else "expired"
    except (ValueError, TypeError) as exc:
        logger.warning("Cannot parse horizon_at %r: %s", horizon_at, exc)
        return "expired"


def evaluate_prediction(
    prediction: PredictionRecord,
    actual_price: Optional[float],
    price_at_generation: Optional[float] = None,
    tolerance_pct: float = 2.0,
) -> AccuracyResult:
    """Evaluate a single prediction against actual market data.

    Parameters
    ----------
    prediction:
        The stored prediction record to evaluate.
    actual_price:
        The actual market price at the prediction's horizon timestamp.
        Pass ``None`` when the price is not yet available.
    price_at_generation:
        The market price at the time the prediction was generated.
        Needed for directional correctness. Falls back to ``predicted_price``
        if not provided.
    tolerance_pct:
        Maximum acceptable percentage drift (default 2%).

    Returns
    -------
    AccuracyResult
        An object containing drift, direction, and status information.
    """
    horizon_status = get_current_status(prediction.horizon_at)

    # --- Case 1: Horizon still in the future ---
    if horizon_status == "pending":
        return AccuracyResult(
            prediction_id=prediction.id,
            pair=prediction.pair,
            predicted_price=prediction.predicted_price,
            actual_price=None,
            percentage_drift=None,
            predicted_direction=prediction.direction,
            actual_direction=None,
            directional_correct=None,
            status=CorrectnessStatus.PENDING,
            tolerance_pct=tolerance_pct,
        )

    # --- Case 2: Horizon has passed but no actual price ---
    if actual_price is None:
        return AccuracyResult(
            prediction_id=prediction.id,
            pair=prediction.pair,
            predicted_price=prediction.predicted_price,
            actual_price=None,
            percentage_drift=None,
            predicted_direction=prediction.direction,
            actual_direction=None,
            directional_correct=None,
            status=CorrectnessStatus.NO_DATA,
            tolerance_pct=tolerance_pct,
        )

    # --- Case 3: We have actual price data — compute everything ---
    drift = calculate_percentage_drift(prediction.predicted_price, actual_price)

    gen_price = price_at_generation if price_at_generation is not None else prediction.predicted_price
    actual_direction = determine_direction(gen_price, actual_price)
    direction_correct = check_directional_correctness(
        prediction.direction, actual_direction
    )

    drift_within_tolerance = drift <= tolerance_pct
    if drift_within_tolerance and direction_correct:
        status = CorrectnessStatus.CORRECT
    else:
        status = CorrectnessStatus.INCORRECT

    return AccuracyResult(
        prediction_id=prediction.id,
        pair=prediction.pair,
        predicted_price=prediction.predicted_price,
        actual_price=actual_price,
        percentage_drift=round(drift, 4),
        predicted_direction=prediction.direction,
        actual_direction=actual_direction,
        directional_correct=direction_correct,
        status=status,
        tolerance_pct=tolerance_pct,
    )


def evaluate_predictions(
    predictions: list[PredictionRecord],
    actual_prices: dict[str, Optional[float]],
    prices_at_generation: Optional[dict[str, Optional[float]]] = None,
    tolerance_pct: float = 2.0,
) -> list[AccuracyResult]:
    """Evaluate a batch of predictions against their actual prices.

    Parameters
    ----------
    predictions:
        List of prediction records to evaluate.
    actual_prices:
        Mapping of ``prediction.id`` → actual price (or ``None``).
    prices_at_generation:
        Optional mapping of ``prediction.id`` → price at generation time.
    tolerance_pct:
        Maximum acceptable percentage drift (default 2%).

    Returns
    -------
    list[AccuracyResult]
        Evaluation results in the same order as *predictions*.
    """
    gen_prices = prices_at_generation or {}
    results: list[AccuracyResult] = []

    for pred in predictions:
        actual = actual_prices.get(pred.id)
        gen_price = gen_prices.get(pred.id) if gen_prices else None
        results.append(
            evaluate_prediction(
                prediction=pred,
                actual_price=actual,
                price_at_generation=gen_price,
                tolerance_pct=tolerance_pct,
            )
        )

    return results
