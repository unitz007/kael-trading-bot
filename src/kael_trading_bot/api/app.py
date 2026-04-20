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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers (borrowed from CLI entry point — kept in sync)
# ---------------------------------------------------------------------------


def _pair_to_model_name(pair: str) -> str:
    """Derive a filesystem-safe model name from a forex pair ticker."""
    return pair.replace("=", "_").replace("^", "_").lower()


def _normalise_ticker(pair: str) -> str:
    """Ensure a pair string has the ``=X`` suffix used by Yahoo Finance."""
    if "=" in pair or "^" in pair:
        return pair
    return f"{pair}=X"


def _is_supported_pair(pair: str) -> bool:
    """Check if *pair* is in the configured default pairs list."""
    ticker = _normalise_ticker(pair)
    return ticker in DEFAULT_FOREX_PAIRS


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
    def train_model(pair: str) -> JSONResponse | dict[str, Any]:
        """Trigger model training for *pair*."""
        ticker = _normalise_ticker(pair)

        if not _is_supported_pair(ticker):
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"Unsupported forex pair: {ticker}",
                    "supported_pairs": DEFAULT_FOREX_PAIRS,
                },
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
            target_col = "target_direction_1"
            if target_col not in feature_df.columns:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": (
                            f"Target column '{target_col}' not found after "
                            f"feature engineering."
                        ),
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
            model_name = _pair_to_model_name(ticker)
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
    def get_predictions(pair: str) -> JSONResponse | dict[str, Any]:
        """Return prediction results for *pair* using the latest trained model."""
        ticker = _normalise_ticker(pair)
        model_name = _pair_to_model_name(ticker)
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
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    prob_up = y_proba[:, 1].tolist()
                else:
                    prob_up = None
            except Exception:
                prob_up = None

            dates = [str(d) for d in feature_df.index]
            label_map = {0.0: "DOWN", 1.0: "UP", 0: "DOWN", 1: "UP"}

            predictions = []
            for i in range(len(dates)):
                pred: dict[str, Any] = {
                    "date": dates[i],
                    "prediction": label_map.get(y_pred[i], str(y_pred[i])),
                }
                if prob_up is not None:
                    pred["probability_up"] = round(float(prob_up[i]), 4)
                predictions.append(pred)

            up_count = int(sum(1 for p in y_pred if p == 1))
            down_count = len(y_pred) - up_count

            return {
                "pair": ticker,
                "model_name": model_name,
                "model_version": version,
                "trained_at": metadata.trained_at,
                "total_predictions": len(predictions),
                "up_count": up_count,
                "down_count": down_count,
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

    return app
