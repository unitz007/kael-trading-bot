"""REST API application factory.

Creates and configures the Flask application that exposes bot
capabilities (forex pair data, model training, predictions) as HTTP
endpoints.

Usage
-----
::

    from kael_trading_bot.api import create_app
    app = create_app()
    app.run(host="0.0.0.0", port=5000)

Or via CLI::

    python main.py serve
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from flask import Flask, Response, jsonify, request

from kael_trading_bot.config import (
    DEFAULT_FOREX_PAIRS,
    IngestionConfig,
)
from kael_trading_bot.features.pipeline import FeatureConfig, build_feature_matrix
from kael_trading_bot.ingestion import ForexDataFetcher
from kael_trading_bot.training.persistence import ModelMetadata, ModelPersistence
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


def create_app() -> Flask:
    """Create and configure the Flask application.

    Returns
    -------
    Flask
        The configured application ready to serve.
    """
    app = Flask(__name__)

    # Allow cross-origin requests so a web frontend on a different port
    # can reach the API during development.
    @app.after_request
    def _add_cors_headers(response: Response) -> Response:
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = (
            "GET, POST, OPTIONS"
        )
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    # Handle preflight OPTIONS requests
    @app.before_request
    def _handle_options():
        if request.method == "OPTIONS":
            response = Response(status=204)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = (
                "GET, POST, OPTIONS"
            )
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            return response

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.route("/api/v1/pairs", methods=["GET"])
    def list_pairs() -> tuple[Response, int]:
        """Return the list of available/supported forex pairs.

        Acceptance Criteria: #2 — GET endpoint returns list of pairs.
        """
        return jsonify(
            {
                "pairs": DEFAULT_FOREX_PAIRS,
                "count": len(DEFAULT_FOREX_PAIRS),
            }
        ), 200

    @app.route("/api/v1/pairs/<pair>/history", methods=["GET"])
    def get_history(pair: str) -> tuple[Response, int]:
        """Return historical OHLCV price data for *pair*.

        Acceptance Criteria: #3, #7, #8, #9.
        """
        ticker = _normalise_ticker(pair)

        if not _is_supported_pair(ticker):
            return (
                jsonify(
                    {
                        "error": f"Unsupported forex pair: {ticker}",
                        "supported_pairs": DEFAULT_FOREX_PAIRS,
                    }
                ),
                404,
            )

        try:
            ingestion_cfg = IngestionConfig(pairs=(ticker,))
            fetcher = ForexDataFetcher(ingestion_cfg)
            df = fetcher.get(ticker)
        except ValueError as exc:
            return (
                jsonify({"error": str(exc)}),
                404,
            )
        except Exception as exc:
            logger.exception("Failed to fetch history for %s", ticker)
            return (
                jsonify({"error": f"Internal error fetching data: {exc}"}),
                500,
            )

        # Convert DataFrame to list of records with ISO date strings
        records = df.reset_index()
        records["Date"] = records["Date"].astype(str)
        data = records.to_dict(orient="records")

        return (
            jsonify(
                {
                    "pair": ticker,
                    "rows": len(data),
                    "data": data,
                }
            ),
            200,
        )

    @app.route("/api/v1/pairs/<pair>/train", methods=["POST"])
    def train_model(pair: str) -> tuple[Response, int]:
        """Trigger model training for *pair*.

        Acceptance Criteria: #4, #7, #8, #9.
        """
        ticker = _normalise_ticker(pair)

        if not _is_supported_pair(ticker):
            return (
                jsonify(
                    {
                        "error": f"Unsupported forex pair: {ticker}",
                        "supported_pairs": DEFAULT_FOREX_PAIRS,
                    }
                ),
                404,
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
                return (
                    jsonify(
                        {
                            "error": (
                                f"Target column '{target_col}' not found after "
                                f"feature engineering."
                            ),
                        }
                    ),
                    400,
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

            return jsonify(response_data), 200

        except ValueError as exc:
            return (
                jsonify({"error": f"Training failed: {exc}"}),
                400,
            )
        except Exception as exc:
            logger.exception("Training failed for %s", ticker)
            return (
                jsonify({"error": f"Internal error during training: {exc}"}),
                500,
            )

    @app.route("/api/v1/pairs/<pair>/predict", methods=["GET"])
    def get_predictions(pair: str) -> tuple[Response, int]:
        """Return prediction results for *pair* using the latest trained model.

        Acceptance Criteria: #5, #7, #8, #9.
        """
        ticker = _normalise_ticker(pair)
        model_name = _pair_to_model_name(ticker)
        persistence = ModelPersistence()

        try:
            versions = persistence.list_versions(model_name)
            if not versions:
                return (
                    jsonify(
                        {
                            "error": (
                                f"No trained model found for pair '{ticker}'. "
                                f"Train a model first via POST /api/v1/pairs/{pair}/train"
                            ),
                        }
                    ),
                    404,
                )
            version = versions[-1]
        except Exception as exc:
            logger.exception("Error listing model versions for %s", model_name)
            return (
                jsonify({"error": f"Internal error listing models: {exc}"}),
                500,
            )

        try:
            model, metadata = persistence.load(model_name, version)
            feature_names = metadata.feature_names

            if not feature_names:
                return (
                    jsonify(
                        {
                            "error": (
                                f"Model '{model_name}' has no feature names "
                                f"recorded. Cannot generate predictions."
                            ),
                        }
                    ),
                    400,
                )

            # Ingest & engineer features
            ingestion_cfg = IngestionConfig(pairs=(ticker,))
            fetcher = ForexDataFetcher(ingestion_cfg)
            raw_df = fetcher.get(ticker)
            raw_df.columns = [c.lower() for c in raw_df.columns]
            feature_df = build_feature_matrix(raw_df, config=FeatureConfig())

            missing_features = [f for f in feature_names if f not in feature_df.columns]
            if missing_features:
                return (
                    jsonify(
                        {
                            "error": (
                                f"Missing features for prediction: {missing_features}. "
                                f"Model was trained on a different feature set."
                            ),
                        }
                    ),
                    400,
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

            return (
                jsonify(
                    {
                        "pair": ticker,
                        "model_name": model_name,
                        "model_version": version,
                        "trained_at": metadata.trained_at,
                        "total_predictions": len(predictions),
                        "up_count": up_count,
                        "down_count": down_count,
                        "predictions": predictions,
                    }
                ),
                200,
            )

        except ValueError as exc:
            return (
                jsonify({"error": str(exc)}),
                404,
            )
        except Exception as exc:
            logger.exception("Prediction failed for %s", ticker)
            return (
                jsonify({"error": f"Internal error during prediction: {exc}"}),
                500,
            )

    @app.route("/api/v1/models", methods=["GET"])
    def list_models() -> tuple[Response, int]:
        """Return a list of trained models with their metadata.

        Acceptance Criteria: #6, #9.
        """
        persistence = ModelPersistence()

        try:
            model_names = persistence.list_models()
        except Exception as exc:
            logger.exception("Error listing models")
            return (
                jsonify({"error": f"Internal error listing models: {exc}"}),
                500,
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

        return (
            jsonify(
                {
                    "models": models_info,
                    "count": len(models_info),
                }
            ),
            200,
        )

    return app