"""Trade setup generation from ML model predictions.

Derives actionable trade setups (entry, stop loss, take profit, direction,
confidence) from model predictions and recent OHLCV data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

from kael_trading_bot.trade_setup_pkg.backtest import select_optimal_rr_ratio

logger = logging.getLogger(__name__)


@dataclass
class TradeSetup:
    """An actionable trade setup derived from model predictions.

    Attributes:
        pair:             Normalised forex pair ticker (e.g. ``EURUSD=X``).
        direction:        ``"buy"`` or ``"sell"``.
        entry_price:      Suggested entry price (latest close).
        stop_loss:        Stop-loss price.
        take_profit:      Take-profit price.
        confidence:       Model confidence score between 0 and 1.
        atr:              ATR value used to compute SL/TP.
        rr_ratio:         Risk-to-reward ratio used (dynamically determined).
        rr_backtest_info: Backtest metadata explaining the chosen R:R ratio.
        model_name:       Name of the model used.
        model_version:    Version of the model used.
        generated_at:     ISO-8601 timestamp of when the setup was created.
    """

    pair: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    atr: float
    rr_ratio: float = 2.0
    rr_backtest_info: dict = field(default_factory=dict)
    model_name: str = ""
    model_version: str = ""
    timeframe: str = ""
    generated_at: str = ""

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON responses."""
        return asdict(self)


def generate_trade_setup(
    pair: str,
    model,
    metadata,
    feature_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    model_name: str,
    model_version: str,
    timeframe: str = "",
) -> TradeSetup:
    """Generate a single trade setup from the latest model prediction.

    The R:R ratio is determined dynamically by backtesting historical data
    for the same currency pair and timeframe, selecting the ratio that
    maximises historical performance.  A minimum floor of 1:2 is enforced.

    Args:
        pair:           Normalised forex pair ticker.
        model:          Trained sklearn-compatible model.
        metadata:       Model metadata object (needs ``feature_names``).
        feature_df:     Feature-engineered DataFrame (latest rows).
        ohlcv_df:       Raw OHLCV DataFrame (used for current price + ATR).
        model_name:     Identifier of the model.
        model_version:  Version string of the model.

    Returns:
        A :class:`TradeSetup` instance.

    Raises:
        ValueError: If inputs are insufficient (empty data, missing ATR, etc.).
    """
    # --- Validate inputs ---
    if feature_df.empty:
        raise ValueError("Feature data is empty — insufficient data for trade setup.")
    if ohlcv_df.empty:
        raise ValueError("OHLCV data is empty — insufficient data for trade setup.")
    if metadata.feature_names is None:
        raise ValueError("Model metadata has no feature names — cannot generate setup.")

    # --- Prepare feature row for prediction ---
    feature_cols = metadata.feature_names
    missing = [c for c in feature_cols if c not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing features in data: {missing}")

    latest_features = feature_df[feature_cols].iloc[-1:].values

    # --- Predict ---
    prediction = model.predict(latest_features)[0]
    probas = model.predict_proba(latest_features)[0]

    # prediction: 1 = up, 0 = down
    direction = "buy" if prediction == 1 else "sell"

    # Confidence: probability assigned to the predicted class
    confidence = float(probas[int(prediction)])

    # --- Derive entry / SL / TP from OHLCV ---
    latest_close = float(ohlcv_df["Close"].iloc[-1])

    # Use ATR if available, otherwise fall back to a percentage of price
    atr_col = "atr_14" if "atr_14" in feature_df.columns else None
    if atr_col is not None:
        atr_value = float(feature_df[atr_col].iloc[-1])
    else:
        # Estimate ATR as 1% of price when not available
        atr_value = latest_close * 0.01

    if atr_value <= 0:
        atr_value = latest_close * 0.01

    # --- Dynamic R:R ratio from backtesting ---
    rr_info = select_optimal_rr_ratio(ohlcv_df)
    rr_ratio = rr_info["rr_ratio"]

    # SL is always 1× ATR; TP = ATR × rr_ratio
    sl_distance = atr_value * 1.0
    tp_distance = atr_value * rr_ratio

    if direction == "buy":
        entry_price = latest_close
        stop_loss = round(entry_price - sl_distance, 5)
        take_profit = round(entry_price + tp_distance, 5)
    else:
        entry_price = latest_close
        stop_loss = round(entry_price + sl_distance, 5)
        take_profit = round(entry_price - tp_distance, 5)

    # --- Build result ---
    from datetime import datetime, timezone

    generated_at = datetime.now(timezone.utc).isoformat()

    setup = TradeSetup(
        pair=pair,
        direction=direction,
        entry_price=round(entry_price, 5),
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=round(confidence, 4),
        atr=round(atr_value, 5),
        rr_ratio=rr_ratio,
        rr_backtest_info={
            "backtested": rr_info["backtested"],
            "reason": rr_info["reason"],
            "min_rr": rr_info["min_rr"],
            "candidates": rr_info.get("candidates", []),
        },
        model_name=model_name,
        model_version=model_version,
        timeframe=timeframe,
        generated_at=generated_at,
    )

    logger.info(
        "Trade setup generated: %s %s @ %.5f (SL=%.5f, TP=%.5f, conf=%.2f%%, R:R=1:%.2f)",
        pair,
        direction,
        entry_price,
        stop_loss,
        take_profit,
        confidence * 100,
        rr_ratio,
    )

    return setup