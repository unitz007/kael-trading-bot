"""Prediction service — wraps the training pipeline for web use.

Exposes a thin async-friendly layer that:
1. Checks whether a trained model exists for a given pair.
2. Generates a prediction using the latest saved model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np

from src.kael_trading_bot.config import DEFAULT_FOREX_PAIRS, IngestionConfig
from src.kael_trading_bot.features.pipeline import FeatureConfig, build_feature_matrix
from src.kael_trading_bot.ingestion import ForexDataFetcher
from src.kael_trading_bot.training.persistence import ModelMetadata, ModelPersistence

logger = logging.getLogger(__name__)


def _pair_to_model_name(pair: str) -> str:
    """Derive a filesystem-safe model name from a forex pair ticker."""
    return pair.replace("=", "_").replace("^", "_").lower()


@dataclass
class PredictionResult:
    """Serialisable prediction result."""

    pair: str
    direction: str  # "UP" or "DOWN"
    confidence: float  # probability of the predicted direction
    predicted_return: float | None
    model_version: str
    model_type: str
    trained_at: str
    generated_at: str


@dataclass
class ModelStatus:
    """Serialisable model availability status."""

    pair: str
    model_name: str
    available: bool
    versions: list[str]
    latest_version: str | None
    trained_at: str | None


class PredictionService:
    """Service layer for generating predictions via the web UI."""

    def __init__(self, persistence: ModelPersistence | None = None) -> None:
        self.persistence = persistence or ModelPersistence()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_available_pairs(self) -> list[str]:
        """Return the configured list of forex pairs."""
        return list(DEFAULT_FOREX_PAIRS)

    def get_model_status(self, pair: str) -> ModelStatus:
        """Check whether a trained model exists for *pair*."""
        model_name = _pair_to_model_name(pair)
        versions = self.persistence.list_versions(model_name)

        latest_version: str | None = None
        trained_at: str | None = None
        if versions:
            latest_version = versions[-1]
            try:
                _, meta = self.persistence.load(model_name, latest_version)
                trained_at = meta.trained_at
            except Exception:
                trained_at = None

        return ModelStatus(
            pair=pair,
            model_name=model_name,
            available=len(versions) > 0,
            versions=versions,
            latest_version=latest_version,
            trained_at=trained_at,
        )

    def get_all_model_statuses(self) -> list[ModelStatus]:
        """Return model statuses for all configured pairs."""
        return [self.get_model_status(pair) for pair in self.list_available_pairs()]

    def predict(self, pair: str) -> PredictionResult:
        """Generate a prediction for *pair* using the latest trained model.

        Raises
        ------
        FileNotFoundError
            If no trained model exists for the pair.
        ValueError
            If feature engineering fails.
        """
        model_name = _pair_to_model_name(pair)

        # 1. Load model
        versions = self.persistence.list_versions(model_name)
        if not versions:
            raise FileNotFoundError(
                f"No trained model found for '{pair}'. "
                f"Train a model first using: python main.py train {pair}"
            )

        version = versions[-1]
        model, metadata: ModelMetadata = self.persistence.load(model_name, version)

        feature_names = metadata.feature_names
        if not feature_names:
            raise ValueError(
                f"Model '{model_name}' has no feature names. Cannot predict."
            )

        # 2. Ingest & engineer features
        ingestion_cfg = IngestionConfig(pairs=(pair,))
        fetcher = ForexDataFetcher(ingestion_cfg)
        raw_df = fetcher.get(pair)
        raw_df.columns = [c.lower() for c in raw_df.columns]
        feature_df = build_feature_matrix(raw_df, config=FeatureConfig())

        # Ensure feature alignment
        missing = [f for f in feature_names if f not in feature_df.columns]
        if missing:
            raise ValueError(
                f"Missing features for prediction: {missing}"
            )

        X = feature_df[feature_names].values.astype(np.float64)

        # 3. Predict — use the last row (most recent data point)
        y_pred = model.predict(X)
        last_idx = len(y_pred) - 1

        direction = "UP" if float(y_pred[last_idx]) == 1.0 else "DOWN"

        confidence = 0.0
        try:
            y_proba = model.predict_proba(X)
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                predicted_class = int(y_pred[last_idx])
                confidence = float(y_proba[last_idx, predicted_class])
        except Exception:
            pass

        # Check for predicted return in feature data
        predicted_return: float | None = None
        for col in feature_df.columns:
            if col == "future_return_1":
                predicted_return = float(feature_df[col].iloc[-1])
                break

        return PredictionResult(
            pair=pair,
            direction=direction,
            confidence=confidence,
            predicted_return=predicted_return,
            model_version=version,
            model_type=metadata.model_type,
            trained_at=metadata.trained_at or "unknown",
            generated_at=datetime.now(timezone.utc).isoformat(),
        )