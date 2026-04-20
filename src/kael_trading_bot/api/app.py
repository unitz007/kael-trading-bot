"""REST API application factory.

Creates and configures the FastAPI application that exposes bot
capabilities (forex pair data, model training, predictions) as HTTP
endpoints.

Usage
-----
::

    import uvicorn
    from kael_trading_bot.api import create_app
    uvicorn.run(create_app(), host="0.0.0.0", port=5000)

Or via CLI::

    python main.py serve
"""

from __future__ import annotations

import threading
import logging
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from kael_trading_bot.config import (
    DEFAULT_FOREX_PAIRS,
    IngestionConfig,
)
from kael_trading_bot.features.pipeline import FeatureConfig, build_feature_matrix
from kael_trading_bot.ingestion import ForexDataFetcher
from kael_trading_bot.training.persistence import ModelPersistence
from kael_trading_bot.training.pipeline import PipelineConfig, TrainingPipeline
from kael_trading_bot.trade_setup import generate_trade_setup
import pandas as pd


# Shared scanner instance — lazily created on first use.
_scanner: "TradeSetupScanner | None" = None
_scanner_lock = threading.Lock()




logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers (borrowed from CLI entry point — kept in sync)
# ---------------------------------------------------------------------------





def _normalise_ticker(pair: str) -> str:
    """Ensure a pair string has the ``=X`` suffix used by Yahoo Finance."""
    if "=" in pair or "^" in pair:
        return pair
    return f"{pair}=X"


def _is_supported_pair(pair: str) -> bool:
    """Check if *pair* is in the configured default pairs list."""
    ticker = _normalise_ticker(pair)
    return ticker in DEFAULT_FOREX_PAIRS

SUPPORTED_TIMEFRAMES = ("5m", "15m", "1h", "4h")
DEFAULT_TIMEFRAME = "1h"


def _validate_timeframe(timeframe: str | None) -> str:
    """Return a validated timeframe string, defaulting to ``DEFAULT_TIMEFRAME``."""
    if timeframe is None:
        return DEFAULT_TIMEFRAME
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(
            f"Invalid timeframe '{timeframe}'. "
            f"Supported timeframes: {', '.join(SUPPORTED_TIMEFRAMES)}"
        )
    return timeframe


def _pair_to_model_name(pair: str, timeframe: str | None = None) -> str:
    """Derive a filesystem-safe model name from a forex pair ticker and timeframe."""
    base = pair.replace("=", "_").replace("^", "_").lower()
    if timeframe:
        return f"{base}_{timeframe}"
    return base


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns
    -------
    FastAPI
        The configured application ready to serve.
    """
    app = FastAPI(
        title="Kael Trading Bot API",
        description="REST API for forex trading bot — data, training, predictions",
        version="0.1.0",
    )

    # Allow cross-origin requests so a web frontend on a different port
    # can reach the API during development.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.get("/api/v1/pairs")
    def list_pairs() -> dict[str, Any]:
        """Return the list of available/supported forex pairs."""
        return {
            "pairs": DEFAULT_FOREX_PAIRS,
            "count": len(DEFAULT_FOREX_PAIRS),
        }

    @app.get("/api/v1/pairs/{pair}/history", response_model=None)
    def get_history(pair: str) -> JSONResponse | dict[str, Any]:
        """Return historical OHLCV price data for *pair*."""
        ticker = _normalise_ticker(pair)

        if not _is_supported_pair(ticker):
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"Unsupported forex pair: {ticker}",
                    "supported_pairs": DEFAULT_FOREX_PAIRS,
                },
            )

        try:
            ingestion_cfg = IngestionConfig(pairs=(ticker,))
            fetcher = ForexDataFetcher(ingestion_cfg)
            df = fetcher.get(ticker)
        except ValueError as exc:
            return JSONResponse(
                status_code=404,
                content={"error": str(exc)},
            )
        except Exception as exc:
            logger.exception("Failed to fetch history for %s", ticker)
            return JSONResponse(
                status_code=500,
                content={"error": f"Internal error fetching data: {exc}"},
            )

        # Convert DataFrame to list of records with ISO date strings
        records = df.reset_index()
        records["Date"] = records["Date"].astype(str)
        data = records.to_dict(orient="records")

        return {
            "pair": ticker,
            "rows": len(data),
            "data": data,
        }

    @app.post("/api/v1/pairs/{pair}/train", response_model=None)
    def train_model(pair: str, timeframe: str | None = None) -> JSONResponse | dict[str, Any]:
        """Trigger model training for *pair* with an optional *timeframe*."""
        ticker = _normalise_ticker(pair)

        if not _is_supported_pair(ticker):
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"Unsupported forex pair: {ticker}",
                    "supported_pairs": DEFAULT_FOREX_PAIRS,
                },
            )

        try:
            tf = _validate_timeframe(timeframe)
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": str(exc)},
            )

        start_time = time.time()

        try:
            # 1. Ingestion
            ingestion_cfg = IngestionConfig(pairs=(ticker,))
            fetcher = ForexDataFetcher(ingestion_cfg)
            raw_df = fetcher.get(ticker)

            # 2. Feature engineering
            raw_df.columns = [c.lower() for c in raw_df.columns]
            feature_df = build_feature_matrix(raw_df, config=FeatureConfig())

            # 3. Prepare arrays
            target_col_candidates = ("target_dir_1", "target_direction_1")
            target_col = next((c for c in target_col_candidates if c in feature_df.columns), None)
            if target_col is None:
                available_targets = sorted(c for c in feature_df.columns if c.startswith("target_"))
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": (
                            f"Target column not found after feature engineering. "
                            f"Tried {list(target_col_candidates)}."
                        ),
                        "available_target_columns": available_targets,
                    },
                )

            exclude_cols = {target_col}
            for col in feature_df.columns:
                if col.startswith("future_return") or col.startswith("target_"):
                    exclude_cols.add(col)

            feature_cols = [c for c in feature_df.columns if c not in exclude_cols]
            X = feature_df[feature_cols].values.astype(np.float64)
            y = feature_df[target_col].values.astype(np.float64)

            # 4. Train
            model_name = _pair_to_model_name(ticker, tf)
            model_version = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%S")

            pipeline = TrainingPipeline(
                config=PipelineConfig(
                    model_type="xgboost",
                    model_name=model_name,
                    model_version=model_version,
                    save_model=True,
                    log_run=True,
                    cross_validate=True,
                ),
            )

            result = pipeline.run(X, y, feature_names=feature_cols)

            duration = time.time() - start_time

            response_data: dict[str, Any] = {
                "pair": ticker,
                "timeframe": tf,
                "model_name": result.model_name,
                "model_version": result.model_version,
                "model_type": result.model_type,
                "status": "completed",
                "duration_seconds": round(duration, 2),
                "samples_trained": int(len(X)),
                "num_features": len(feature_cols),
            }

            if result.test_eval is not None:
                response_data["test_metrics"] = result.test_eval.to_dict()

            if result.saved_path:
                response_data["saved_path"] = result.saved_path

            return response_data

        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": f"Training failed: {exc}"},
            )
        except Exception as exc:
            logger.exception("Training failed for %s", ticker)
            return JSONResponse(
                status_code=500,
                content={"error": f"Internal error during training: {exc}"},
            )

    @app.get("/api/v1/pairs/{pair}/predict", response_model=None)
    def get_predictions(pair: str, timeframe: str | None = None) -> JSONResponse | dict[str, Any]:
        """Return prediction results for *pair* using the latest trained model."""
        ticker = _normalise_ticker(pair)

        try:
            tf = _validate_timeframe(timeframe)
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": str(exc)},
            )

        model_name = _pair_to_model_name(ticker, tf)
        persistence = ModelPersistence()

        try:
            versions = persistence.list_versions(model_name)
            if not versions:
                return JSONResponse(
                    status_code=404,
                    content={
                        "error": (
                            f"No trained model found for pair '{ticker}'. "
                            f"Train a model first via POST /api/v1/pairs/{pair}/train"
                        ),
                    },
                )
            version = versions[-1]
        except Exception as exc:
            logger.exception("Error listing model versions for %s", model_name)
            return JSONResponse(
                status_code=500,
                content={"error": f"Internal error listing models: {exc}"},
            )

        try:
            model, metadata = persistence.load(model_name, version)
            feature_names = metadata.feature_names

            if not feature_names:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": (
                            f"Model '{model_name}' has no feature names "
                            f"recorded. Cannot generate predictions."
                        ),
                    },
                )

            # Ingest & engineer features
            ingestion_cfg = IngestionConfig(pairs=(ticker,))
            fetcher = ForexDataFetcher(ingestion_cfg)
            raw_df = fetcher.get(ticker)
            raw_df.columns = [c.lower() for c in raw_df.columns]
            feature_df = build_feature_matrix(raw_df, config=FeatureConfig())

            missing_features = [f for f in feature_names if f not in feature_df.columns]
            if missing_features:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": (
                            f"Missing features for prediction: {missing_features}. "
                            f"Model was trained on a different feature set."
                        ),
                    },
                )

            X = feature_df[feature_names].values.astype(np.float64)
            y_pred = model.predict(X)

            try:
                y_proba = model.predict_proba(X)
                prob_up = None
                if y_proba.ndim == 2:
                    label_values = getattr(metadata, "label_values", None)
                    if label_values:
                        try:
                            up_idx = list(label_values).index(1)
                            if up_idx < y_proba.shape[1]:
                                prob_up = y_proba[:, up_idx].tolist()
                        except ValueError:
                            prob_up = None
                    elif y_proba.shape[1] == 2:
                        prob_up = y_proba[:, 1].tolist()
            except Exception:
                prob_up = None

            dates = [str(d) for d in feature_df.index]
            label_values = getattr(metadata, "label_values", None)
            if label_values:
                label_map = {float(v): str(v) for v in label_values}
                label_map.update({-1.0: "DOWN", 0.0: "FLAT", 1.0: "UP", -1: "DOWN", 0: "FLAT", 1: "UP"})
            else:
                # Backward-compatible fallback for older models without metadata.label_values.
                label_map = {
                    -1.0: "DOWN",
                    0.0: "FLAT",
                    1.0: "UP",
                    -1: "DOWN",
                    0: "FLAT",
                    1: "UP",
                    0.0: "DOWN",
                    1.0: "UP",
                    2.0: "UP",
                    0: "DOWN",
                    1: "FLAT",
                    2: "UP",
                }

            # Decode model outputs if they look like encoded class indices.
            if label_values:
                y_pred_arr = np.asarray(y_pred)
                if y_pred_arr.ndim == 1 and np.issubdtype(y_pred_arr.dtype, np.integer):
                    if y_pred_arr.size > 0 and y_pred_arr.min() >= 0 and y_pred_arr.max() < len(label_values):
                        y_pred = np.asarray(label_values, dtype=float)[y_pred_arr]

            predictions = []
            for i in range(len(dates)):
                pred: dict[str, Any] = {
                    "date": dates[i],
                    "prediction": label_map.get(y_pred[i], str(y_pred[i])),
                }
                if prob_up is not None:
                    pred["probability_up"] = round(float(prob_up[i]), 4)
                predictions.append(pred)

            y_pred_list = list(y_pred)
            up_count = int(sum(1 for p in y_pred_list if float(p) == 1.0))
            down_count = int(sum(1 for p in y_pred_list if float(p) == -1.0))
            flat_count = int(sum(1 for p in y_pred_list if float(p) == 0.0))

            return {
                "pair": ticker,
                "timeframe": tf,
                "model_name": model_name,
                "model_version": version,
                "trained_at": metadata.trained_at,
                "total_predictions": len(predictions),
                "up_count": up_count,
                "down_count": down_count,
                "flat_count": flat_count,
                "predictions": predictions,
            }

        except ValueError as exc:
            return JSONResponse(
                status_code=404,
                content={"error": str(exc)},
            )
        except Exception as exc:
            logger.exception("Prediction failed for %s", ticker)
            return JSONResponse(
                status_code=500,
                content={"error": f"Internal error during prediction: {exc}"},
            )

    @app.get("/api/v1/pairs/{pair}/forecast", response_model=None)
    def get_forecast(pair: str, horizon: int = 30, timeframe: str | None = None) -> JSONResponse | dict[str, Any]:
        """Return future price forecast for *pair* using the latest trained model.

        Parameters
        ----------
        pair:
            Forex pair ticker.
        horizon:
            Number of future periods to forecast.
        timeframe:
            Time frame for the model (5m, 15m, 1h, 4h).
        """
        ticker = _normalise_ticker(pair)

        try:
            tf = _validate_timeframe(timeframe)
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": str(exc)},
            )

        model_name = _pair_to_model_name(ticker, tf)
        persistence = ModelPersistence()

        # Validate horizon
        if horizon < 1 or horizon > 365:
            return JSONResponse(
                status_code=400,
                content={"error": "horizon must be between 1 and 365"},
            )

        # Check for trained model
        try:
            versions = persistence.list_versions(model_name)
            if not versions:
                return JSONResponse(
                    status_code=404,
                    content={
                        "error": (
                            f"No trained model found for pair '{ticker}'. "
                            f"Train a model first via POST /api/v1/pairs/{pair}/train"
                        ),
                    },
                )
            version = versions[-1]
        except Exception as exc:
            logger.exception("Error listing model versions for %s", model_name)
            return JSONResponse(
                status_code=500,
                content={"error": f"Internal error listing models: {exc}"},
            )

        try:
            model, metadata = persistence.load(model_name, version)
            feature_names = metadata.feature_names

            if not feature_names:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": (
                            f"Model '{model_name}' has no feature names "
                            f"recorded. Cannot generate forecast."
                        ),
                    },
                )

            # Ingest & engineer features
            ingestion_cfg = IngestionConfig(pairs=(ticker,))
            fetcher = ForexDataFetcher(ingestion_cfg)
            raw_df = fetcher.get(ticker)
            raw_df.columns = [c.lower() for c in raw_df.columns]
            feature_df = build_feature_matrix(raw_df, config=FeatureConfig())

            missing_features = [f for f in feature_names if f not in feature_df.columns]
            if missing_features:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": (
                            f"Missing features for forecast: {missing_features}. "
                            f"Model was trained on a different feature set."
                        ),
                    },
                )

            # Use the label encoding from training time
            label_values = getattr(metadata, "label_values", None)

            # Recursive multi-step forecast
            last_row = feature_df[feature_names].iloc[-1:].copy()
            last_close = float(raw_df["close"].iloc[-1])
            last_date = feature_df.index[-1]

            # Determine step frequency from index
            if len(feature_df.index) >= 2:
                freq = feature_df.index[-1] - feature_df.index[-2]
            else:
                freq = pd.Timedelta(days=1)

            # Compute prediction confidence from last training probabilities
            last_features = last_row.values.astype(np.float64)
            try:
                last_proba = model.predict_proba(last_features)
                label_map = {-1.0: "DOWN", 0.0: "FLAT", 1.0: "UP"}
                if label_values:
                    max_prob = float(np.max(last_proba[0]))
                else:
                    max_prob = float(np.max(last_proba[0]))
            except Exception:
                max_prob = 0.5

            # Estimate ATR-based confidence band from recent volatility
            if "atr" in raw_df.columns:
                recent_atr = float(raw_df["atr"].iloc[-5:].mean())
            else:
                returns = raw_df["close"].pct_change().dropna()
                recent_atr = float(returns.iloc[-5:].std() * last_close) if len(returns) >= 5 else last_close * 0.01

            forecast_prices = []
            current_price = last_close
            direction = None

            # Direction from initial prediction
            last_X = feature_df[feature_names].iloc[-1:].values.astype(np.float64)
            try:
                pred_encoded = model.predict(last_X)[0]
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
            except Exception:
                direction = "FLAT"

            for step in range(1, horizon + 1):
                future_date = last_date + freq * step

                # Confidence band widens with sqrt of time step
                band_width = recent_atr * 1.5 * (step ** 0.5)

                if direction == "UP":
                    change = band_width * 0.15 * step
                elif direction == "DOWN":
                    change = -band_width * 0.15 * step
                else:
                    change = 0

                predicted_price = current_price + change
                upper = predicted_price + band_width
                lower = predicted_price - band_width

                # Trend back toward mean for very long horizons
                if step > horizon * 0.5:
                    mean_reversion = (last_close - predicted_price) * 0.02
                    predicted_price += mean_reversion
                    upper += mean_reversion
                    lower += mean_reversion

                current_price = predicted_price

                forecast_prices.append({
                    "date": str(future_date),
                    "predicted_price": round(float(predicted_price), 5),
                    "upper_bound": round(float(upper), 5),
                    "lower_bound": round(float(lower), 5),
                    "direction": direction,
                    "confidence": round(float(max_prob), 4),
                })

            # Historical data for chart context
            hist_records = raw_df["close"].tail(90).reset_index()
            hist_records.columns = ["date", "close"]
            hist_records["date"] = hist_records["date"].astype(str)
            hist_records["close"] = hist_records["close"].astype(float)
            historical_data = hist_records.to_dict(orient="records")

            return {
                "pair": ticker,
                "timeframe": tf,
                "model_name": model_name,
                "model_version": version,
                "trained_at": metadata.trained_at,
                "forecast_horizon": horizon,
                "last_historical_date": str(last_date),
                "last_historical_price": last_close,
                "direction": direction,
                "confidence": round(float(max_prob), 4),
                "historical_data": historical_data,
                "forecast": forecast_prices,
            }

        except ValueError as exc:
            return JSONResponse(
                status_code=404,
                content={"error": str(exc)},
            )
        except Exception as exc:
            logger.exception("Forecast failed for %s", ticker)
            return JSONResponse(
                status_code=500,
                content={"error": f"Internal error during forecast: {exc}"},
            )

    @app.get("/api/v1/pairs/{pair}/trade-setup", response_model=None)
    def get_trade_setup(pair: str, timeframe: str | None = None) -> JSONResponse | dict[str, Any]:
        """Generate an actionable trade setup for *pair* using the latest trained model.

        Returns entry price, stop loss, take profit, model confidence,
        and trade direction (buy/sell).
        """
        ticker = _normalise_ticker(pair)

        if not _is_supported_pair(ticker):
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"Unsupported forex pair: {ticker}",
                    "supported_pairs": DEFAULT_FOREX_PAIRS,
                },
            )

        try:
            tf = _validate_timeframe(timeframe)
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": str(exc)},
            )

        model_name = _pair_to_model_name(ticker, tf)
        persistence = ModelPersistence()

        try:
            versions = persistence.list_versions(model_name)
            if not versions:
                return JSONResponse(
                    status_code=404,
                    content={
                        "error": (
                            f"No trained model found for pair '{ticker}'. "
                            f"Train a model first via POST /api/v1/pairs/{pair}/train"
                        ),
                    },
                )
            version = versions[-1]
        except Exception as exc:
            logger.exception("Error listing model versions for %s", model_name)
            return JSONResponse(
                status_code=500,
                content={"error": f"Internal error listing models: {exc}"},
            )

        try:
            model, metadata = persistence.load(model_name, version)

            if not metadata.feature_names:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": (
                            f"Model '{model_name}' has no feature names "
                            f"recorded. Cannot generate trade setup."
                        ),
                    },
                )

            # Ingest & engineer features
            ingestion_cfg = IngestionConfig(pairs=(ticker,))
            fetcher = ForexDataFetcher(ingestion_cfg)
            raw_df = fetcher.get(ticker)
            raw_df_copy = raw_df.copy()
            raw_df.columns = [c.lower() for c in raw_df.columns]
            feature_df = build_feature_matrix(raw_df, config=FeatureConfig())

            missing_features = [
                f for f in metadata.feature_names if f not in feature_df.columns
            ]
            if missing_features:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": (
                            f"Missing features for trade setup: {missing_features}. "
                            f"Model was trained on a different feature set."
                        ),
                    },
                )

            setup = generate_trade_setup(
                pair=ticker,
                model=model,
                metadata=metadata,
                feature_df=feature_df,
                ohlcv_df=raw_df_copy,
                model_name=model_name,
                model_version=version,
                timeframe=tf,
            )

            return setup.to_dict()

        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": f"Could not generate trade setup: {exc}"},
            )
        except Exception as exc:
            logger.exception("Trade setup generation failed for %s", ticker)
            return JSONResponse(
                status_code=500,
                content={"error": f"Internal error generating trade setup: {exc}"},
            )

    @app.get("/api/v1/models", response_model=None)
    def list_models() -> JSONResponse | dict[str, Any]:
        """Return a list of trained models with their metadata."""
        persistence = ModelPersistence()

        try:
            model_names = persistence.list_models()
        except Exception as exc:
            logger.exception("Error listing models")
            return JSONResponse(
                status_code=500,
                content={"error": f"Internal error listing models: {exc}"},
            )

        models_info: list[dict[str, Any]] = []

        for name in model_names:
            versions = persistence.list_versions(name)
            # Get metadata for each version
            for version in versions:
                try:
                    _, metadata = persistence.load(name, version)
                    models_info.append(
                        {
                            "model_name": name,
                            "version": version,
                            "model_type": metadata.model_type,
                            "trained_at": metadata.trained_at,
                            "metrics": metadata.metrics,
                            "feature_names": metadata.feature_names,
                        }
                    )
                except Exception:
                    # If a version's metadata is corrupted, skip it
                    logger.warning(
                        "Could not load metadata for model %s version %s",
                        name,
                        version,
                    )

        return {
            "models": models_info,
            "count": len(models_info),
        }


    # ------------------------------------------------------------------
    # Scanner query endpoint
    # ------------------------------------------------------------------

    def _get_scanner() -> "TradeSetupScanner":
        """Return (or create) the shared scanner instance."""
        global _scanner
        if _scanner is None:
            with _scanner_lock:
                if _scanner is None:
                    from kael_trading_bot.config import ScannerConfig
                    from kael_trading_bot.scanner.scheduler import TradeSetupScanner

                    _scanner = TradeSetupScanner(ScannerConfig())
        return _scanner

    @app.get("/api/v1/setups", response_model=None)
    def list_setups(
        pair: str | None = None,
        timeframe: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Return the most recent scanned trade setups.

        Results are queryable by *pair* and/or *timeframe*.
        """
        scanner = _get_scanner()
        setups = scanner.store.query(pair=pair, timeframe=timeframe, limit=limit)
        return {
            "setups": setups,
            "count": len(setups),
            "pair": pair,
            "timeframe": timeframe,
        }


    return app