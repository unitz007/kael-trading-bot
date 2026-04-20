"""Kael Trading Bot — CLI entry point.

Provides three commands:

* **train** — ingest data, engineer features, train an ML model, and
  persist the trained model to disk.
* **predict** — load a persisted model and output predictions for the
  configured forex pair.
* **serve** — start the REST API server exposing bot capabilities.

Usage examples
--------------
::

    python main.py train EURUSD
    python main.py predict EURUSD
    python main.py serve --port 5000
    python main.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from kael_trading_bot.config import IngestionConfig
from kael_trading_bot.features.pipeline import FeatureConfig, build_feature_matrix
from kael_trading_bot.ingestion import ForexDataFetcher
from kael_trading_bot.training.persistence import ModelPersistence
from kael_trading_bot.training.pipeline import PipelineConfig, TrainingPipeline

# NOTE: Flask import is deferred (inside cmd_serve) to keep
# `python main.py train/predict` fast when the API dependency
# is not installed.

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kael_trading_bot")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pair_to_model_name(pair: str) -> str:
    """Derive a filesystem-safe model name from a forex pair ticker."""
    return pair.replace("=", "_").replace("^", "_").lower()


def _latest_version(persistence: ModelPersistence, model_name: str) -> str:
    """Return the most recent saved version for *model_name*.

    Raises ``FileNotFoundError`` when no versions exist.
    """
    versions = persistence.list_versions(model_name)
    if not versions:
        raise FileNotFoundError(
            f"No saved model found for '{model_name}'. "
            f"Run 'python main.py train {model_name}' first."
        )
    return versions[-1]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_train(pair: str) -> None:
    """Run the full training pipeline for *pair*."""
    ticker = pair if "=" in pair or "^" in pair else f"{pair}=X"

    logger.info("=" * 60)
    logger.info("Training pipeline — pair: %s", ticker)
    logger.info("=" * 60)

    # -- 1. Ingestion -------------------------------------------------------
    logger.info("[1/4] Ingesting data for %s …", ticker)
    ingestion_cfg = IngestionConfig(pairs=(ticker,))
    fetcher = ForexDataFetcher(ingestion_cfg)
    raw_df = fetcher.get(ticker)
    logger.info("  ✓ Ingestion complete — %d rows fetched.", len(raw_df))

    # -- 2. Feature engineering ---------------------------------------------
    logger.info("[2/4] Engineering features …")
    # Ensure columns are lower-case for the feature pipeline
    raw_df.columns = [c.lower() for c in raw_df.columns]
    feature_df = build_feature_matrix(raw_df, config=FeatureConfig())
    logger.info("  ✓ Feature engineering complete — %d rows, %d columns.", len(feature_df), len(feature_df.columns))

    # -- 3. Prepare arrays for training -------------------------------------
    logger.info("[3/4] Preparing training data …")

    # Identify feature vs target columns
    target_col = "target_direction_1"  # 1-step-ahead direction
    if target_col not in feature_df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {list(feature_df.columns)}"
        )

    # Drop non-feature columns
    exclude_cols = {target_col}
    # Also exclude other target/return columns
    for col in feature_df.columns:
        if col.startswith("future_return") or col.startswith("target_"):
            exclude_cols.add(col)

    feature_cols = [c for c in feature_df.columns if c not in exclude_cols]

    X = feature_df[feature_cols].values.astype(np.float64)
    y = feature_df[target_col].values.astype(np.float64)

    logger.info("  ✓ %d samples, %d features, target: %s", len(X), len(feature_cols), target_col)

    # -- 4. Train model -----------------------------------------------------
    logger.info("[4/4] Training model …")
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

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("  Model name    : %s", result.model_name)
    logger.info("  Model version : %s", result.model_version)
    logger.info("  Duration      : %.1f seconds", result.duration_seconds)
    if result.test_eval is not None:
        logger.info(
            "  Test F1       : %.4f",
            result.test_eval.classification.f1,
        )
    if result.saved_path:
        logger.info("  Saved to      : %s", result.saved_path)
    logger.info("=" * 60)


def cmd_predict(pair: str) -> None:
    """Load a trained model and output predictions for *pair*."""
    ticker = pair if "=" in pair or "^" in pair else f"{pair}=X"

    logger.info("=" * 60)
    logger.info("Prediction pipeline — pair: %s", ticker)
    logger.info("=" * 60)

    model_name = _pair_to_model_name(ticker)
    persistence = ModelPersistence()

    # -- 1. Load model ------------------------------------------------------
    logger.info("[1/3] Loading trained model for %s …", model_name)
    version = _latest_version(persistence, model_name)
    model, metadata = persistence.load(model_name, version)
    logger.info("  ✓ Model loaded — version %s (trained %s)", version, metadata.trained_at)

    feature_names = metadata.feature_names
    if not feature_names:
        raise ValueError(
            f"Model '{model_name}' has no feature names recorded in metadata. "
            f"Cannot build feature matrix for prediction."
        )

    # -- 2. Ingest & engineer features --------------------------------------
    logger.info("[2/3] Ingesting & engineering features for %s …", ticker)
    ingestion_cfg = IngestionConfig(pairs=(ticker,))
    fetcher = ForexDataFetcher(ingestion_cfg)
    raw_df = fetcher.get(ticker)
    raw_df.columns = [c.lower() for c in raw_df.columns]
    feature_df = build_feature_matrix(raw_df, config=FeatureConfig())
    logger.info("  ✓ %d rows, %d columns.", len(feature_df), len(feature_df.columns))

    # Use only the features the model was trained on
    missing_features = [f for f in feature_names if f not in feature_df.columns]
    if missing_features:
        raise ValueError(
            f"Missing features for prediction: {missing_features}. "
            f"Model was trained on a different feature set."
        )

    X = feature_df[feature_names].values.astype(np.float64)

    # -- 3. Predict ---------------------------------------------------------
    logger.info("[3/3] Generating predictions …")
    y_pred = model.predict(X)

    try:
        y_proba = model.predict_proba(X)
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            prob_up = y_proba[:, 1]
        else:
            prob_up = None
    except Exception:
        prob_up = None

    # Print results table
    dates = feature_df.index
    label_map = {0.0: "DOWN", 1.0: "UP", 0: "DOWN", 1: "UP"}

    print()
    print(f"{'Date':<12} {'Prediction':<12} {'Probability (UP)':<18}")
    print("-" * 42)

    # Show the last 20 predictions for readability
    n_show = min(20, len(dates))
    start_idx = len(dates) - n_show
    for i in range(start_idx, len(dates)):
        date_str = pd.Timestamp(dates[i]).strftime("%Y-%m-%d")
        pred_label = label_map.get(y_pred[i], str(y_pred[i]))
        prob_str = f"{prob_up[i]:.4f}" if prob_up is not None else "N/A"
        print(f"{date_str:<12} {pred_label:<12} {prob_str:<18}")

    if n_show < len(dates):
        print(f"… ({len(dates) - n_show} earlier rows omitted)")

    print()

    # Summary
    up_count = int(sum(1 for p in y_pred if p == 1))
    down_count = len(y_pred) - up_count
    logger.info("Predictions: %d UP, %d DOWN (total %d)", up_count, down_count, len(y_pred))
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def cmd_serve(port: int) -> None:
    """Start the REST API server."""
    import os

    os.environ.setdefault("FLASK_APP", "kael_trading_bot.api")
    os.environ.setdefault("KAEL_API_PORT", str(port))

    from kael_trading_bot.api import create_app

    logger.info("Starting API server on port %d", port)
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=False)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Kael Trading Bot — train ML models on forex pairs and "
            "generate predictions.\n\n"
            "Commands:\n"
            "  train   Run the full training pipeline for a forex pair.\n"
            "  predict Load a trained model and output predictions.\n"
            "  serve   Start the REST API server.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # train
    train_parser = subparsers.add_parser(
        "train",
        help="Train an ML model for a forex pair.",
        description=(
            "Ingest historical data, engineer features, and train an "
            "XGBoost classification model for the given forex pair. "
            "The trained model is persisted to disk under the 'models/' "
            "directory."
        ),
    )
    train_parser.add_argument(
        "pair",
        help=(
            "Forex pair to train on, e.g. EURUSD, GBPUSD. "
            "The '=X' suffix is added automatically if omitted."
        ),
    )

    # predict
    predict_parser = subparsers.add_parser(
        "predict",
        help="Generate predictions using a trained model.",
        description=(
            "Load the most recently trained model for the given forex pair, "
            "fetch the latest data, engineer features, and output direction "
            "predictions."
        ),
    )
    predict_parser.add_argument(
        "pair",
        help=(
            "Forex pair to predict on, e.g. EURUSD, GBPUSD. "
            "The '=X' suffix is added automatically if omitted."
        ),
    )

    # serve
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the REST API server.",
        description=(
            "Start a Flask-based REST API that exposes bot capabilities "
            "(forex pairs, historical data, model training, predictions) "
            "as HTTP endpoints."
        ),
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to listen on (default: 5000).",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "train":
        cmd_train(args.pair)
    elif args.command == "predict":
        cmd_predict(args.pair)
    elif args.command == "serve":
        cmd_serve(args.port)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()