"""WebSocket endpoint for live forecast and price streaming.

Provides a WebSocket at ``/ws/chart`` that pushes real-time price
updates and forecast comparison data to connected clients.  The
client can subscribe to a specific *pair* and *timeframe* via query
parameters.

The endpoint periodically fetches the latest live price from Yahoo
Finance, compares it against the model's last forecast, and pushes
a comparison payload to every connected subscriber.

Dependencies (deferred import so the rest of the API works without
``websockets`` installed — FastAPI will raise a clear error at
startup if the package is missing).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

logger = logging.getLogger(__name__)

# Default interval between live-data pushes (seconds).
PUSH_INTERVAL: float = 5.0

# How many recent data points to keep in the rolling buffer.
MAX_BUFFER_SIZE: int = 200

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_ticker(pair: str) -> str:
    """Ensure a pair string has the ``=X`` suffix used by Yahoo Finance."""
    if "=" in pair or "^" in pair:
        return pair
    return f"{pair}=X"


def _validate_timeframe(timeframe: str | None) -> str:
    """Return a validated timeframe string, defaulting to ``1h``."""
    supported = ("5m", "15m", "1h", "4h")
    if timeframe is None:
        return "1h"
    if timeframe not in supported:
        raise ValueError(
            f"Invalid timeframe '{timeframe}'. "
            f"Supported: {', '.join(supported)}"
        )
    return timeframe


def _pair_to_model_name(pair: str, timeframe: str | None = None) -> str:
    """Derive a filesystem-safe model name from a forex pair ticker and timeframe."""
    base = pair.replace("=", "_").replace("^", "_").lower()
    if timeframe:
        return f"{base}_{timeframe}"
    return base


def _fetch_live_price(ticker: str) -> float | None:
    """Fetch the latest closing price for *ticker* from Yahoo Finance.

    Returns ``None`` when the fetch fails (network error, bad ticker, etc.)
    so the WebSocket loop can gracefully skip a cycle.
    """
    try:
        from kael_trading_bot.config import IngestionConfig
        from kael_trading_bot.ingestion import ForexDataFetcher

        cfg = IngestionConfig(pairs=(ticker,))
        fetcher = ForexDataFetcher(cfg)
        df = fetcher.get(ticker)
        if df is not None and not df.empty:
            last_close = float(df["close"].iloc[-1])
            last_date = str(df.index[-1])
            return last_close
        return None
    except Exception:
        logger.warning("Failed to fetch live price for %s", ticker, exc_info=True)
        return None


def _fetch_forecast(pair: str, timeframe: str) -> dict[str, Any] | None:
    """Fetch the latest forecast for *pair* / *timeframe*.

    We replicate the lightest possible version of the forecast endpoint
    to get just the forecasted prices — no need for the full endpoint
    machinery.
    """
    try:
        from kael_trading_bot.config import IngestionConfig
        from kael_trading_bot.features.pipeline import FeatureConfig, build_feature_matrix
        from kael_trading_bot.ingestion import ForexDataFetcher
        from kael_trading_bot.training.persistence import ModelPersistence

        model_name = _pair_to_model_name(pair, timeframe)
        persistence = ModelPersistence()

        versions = persistence.list_versions(model_name)
        if not versions:
            return None
        version = versions[-1]

        model, metadata = persistence.load(model_name, version)
        if not metadata.feature_names:
            return None

        ingestion_cfg = IngestionConfig(pairs=(pair,))
        fetcher = ForexDataFetcher(ingestion_cfg)
        raw_df = fetcher.get(pair)
        raw_df.columns = [c.lower() for c in raw_df.columns]
        feature_df = build_feature_matrix(raw_df, config=FeatureConfig())

        missing = [f for f in metadata.feature_names if f not in feature_df.columns]
        if missing:
            return None

        # Determine direction from last prediction
        last_X = feature_df[metadata.feature_names].iloc[-1:].values
        import numpy as np

        pred_encoded = model.predict(last_X)[0]
        label_values = getattr(metadata, "label_values", None)
        if label_values:
            label_arr = np.asarray(label_values, dtype=float)
            if np.issubdtype(np.asarray(pred_encoded).dtype, np.integer):
                pred_label = label_arr[int(pred_encoded)]
            else:
                pred_label = float(pred_encoded)
        else:
            pred_label = float(pred_encoded)

        if pred_label == 1.0:
            direction = "UP"
        elif pred_label == -1.0:
            direction = "DOWN"
        else:
            direction = "FLAT"

        # Confidence
        try:
            proba = model.predict_proba(last_X)
            confidence = float(np.max(proba[0]))
        except Exception:
            confidence = 0.5

        last_close = float(raw_df["close"].iloc[-1])

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "last_price": last_close,
            "model_name": model_name,
            "model_version": version,
        }
    except Exception:
        logger.warning(
            "Failed to fetch forecast for %s/%s", pair, timeframe, exc_info=True
        )
        return None


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@router.websocket("/ws/chart")
async def websocket_chart(
    websocket: WebSocket,
    pair: str = Query(""),
    timeframe: str = Query("1h"),
):
    """WebSocket that streams live forecast data.

    Query parameters
    ----------------
    pair:
        Forex pair ticker (e.g. ``EURUSD=X``).
    timeframe:
        Timeframe (``5m``, ``15m``, ``1h``, ``4h``).  Defaults to ``1h``.
    """
    await websocket.accept()

    ticker = _normalise_ticker(pair) if pair else ""

    if not ticker:
        await websocket.send_json(
            {
                "type": "error",
                "message": "Query parameter 'pair' is required.",
            }
        )
        await websocket.close(code=1008, reason="Missing pair parameter")
        return

    try:
        _validate_timeframe(timeframe)
    except ValueError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close(code=1008, reason=str(exc))
        return

    logger.info(
        "WebSocket chart client connected: pair=%s timeframe=%s",
        ticker,
        timeframe,
    )

    # Rolling buffer of data points to send to newly connected clients.
    buffer: list[dict[str, Any]] = []
    last_forecast: dict[str, Any] | None = None
    forecast_fetch_counter = 0

    try:
        while True:
            # 1. Fetch live price (run in executor to avoid blocking event loop)
            loop = asyncio.get_running_loop()
            live_price = await loop.run_in_executor(None, _fetch_live_price, ticker)

            # 2. Re-fetch forecast every ~60 seconds (expensive)
            forecast_fetch_counter += 1
            if forecast_fetch_counter >= 12 or last_forecast is None:
                last_forecast = await loop.run_in_executor(
                    None, _fetch_forecast, ticker, timeframe
                )
                forecast_fetch_counter = 0

            now_iso = datetime.now(timezone.utc).isoformat()

            if live_price is not None:
                # Calculate drift from forecast direction
                drift_pct = None
                forecast_price = None
                if last_forecast:
                    forecast_price = last_forecast["last_price"]
                    if forecast_price and forecast_price != 0:
                        drift_pct = round(
                            ((live_price - forecast_price) / forecast_price) * 100, 4
                        )

                point: dict[str, Any] = {
                    "timestamp": now_iso,
                    "live_price": round(live_price, 5),
                    "forecast_price": round(forecast_price, 5) if forecast_price else None,
                    "drift_pct": drift_pct,
                    "pair": ticker,
                    "timeframe": timeframe,
                    "forecast_direction": last_forecast["direction"] if last_forecast else None,
                    "forecast_confidence": last_forecast["confidence"] if last_forecast else None,
                }

                buffer.append(point)
                if len(buffer) > MAX_BUFFER_SIZE:
                    buffer = buffer[-MAX_BUFFER_SIZE:]

                await websocket.send_json({"type": "tick", "data": point})
            else:
                # Still send a heartbeat so the client knows the connection is alive
                await websocket.send_json(
                    {
                        "type": "heartbeat",
                        "timestamp": now_iso,
                        "pair": ticker,
                        "timeframe": timeframe,
                    }
                )

            await asyncio.sleep(PUSH_INTERVAL)

    except WebSocketDisconnect:
        logger.info(
            "WebSocket chart client disconnected: pair=%s timeframe=%s",
            ticker,
            timeframe,
        )
    except Exception:
        logger.exception(
            "Unexpected error in WebSocket chart for %s/%s", ticker, timeframe
        )
        try:
            await websocket.send_json(
                {"type": "error", "message": "Internal server error"}
            )
            await websocket.close(code=1011, reason="Internal server error")
        except Exception:
            pass
