"""Registry of supported ML models.

Provides a thin factory layer so the training pipeline can instantiate
models by type string / enum while keeping hyper-parameter defaults
centralised.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Supported model types."""

    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"


# ---------------------------------------------------------------------------
# Default hyper-parameters per model type
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict[ModelType, dict[str, Any]] = {
    ModelType.XGBOOST: {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": 42,
        "verbosity": 0,
    },
    ModelType.LIGHTGBM: {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    },
    ModelType.LOGISTIC_REGRESSION: {
        "max_iter": 1000,
        "C": 1.0,
        "solver": "lbfgs",
        "random_state": 42,
    },
    ModelType.RANDOM_FOREST: {
        "n_estimators": 200,
        "max_depth": 10,
        "random_state": 42,
        "n_jobs": -1,
    },
}


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Factory for creating model instances.

    Usage::

        model = ModelRegistry.create(ModelType.XGBOOST, n_estimators=500)
    """

    @staticmethod
    def create(model_type: ModelType | str, **overrides: Any) -> Any:
        """Instantiate a model with merged defaults + overrides.

        Parameters
        ----------
        model_type:
            A :class:`ModelType` member or its string value.
        **overrides:
            Hyper-parameter overrides applied on top of defaults.

        Returns
        -------
        A fitted-ready model instance.
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type)

        params = ModelRegistry.default_params(model_type)
        params.update(overrides)

        logger.debug("Creating %s with params %s", model_type.value, params)

        if model_type == ModelType.XGBOOST:
            from xgboost import XGBClassifier
            return XGBClassifier(**params)

        if model_type == ModelType.LIGHTGBM:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**params)

        if model_type == ModelType.LOGISTIC_REGRESSION:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**params)

        if model_type == ModelType.RANDOM_FOREST:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**params)

        raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def default_params(model_type: ModelType | str) -> dict[str, Any]:
        """Return the default hyper-parameters for a model type."""
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        return dict(_DEFAULT_PARAMS[model_type])

    @staticmethod
    def available_models() -> list[str]:
        """Return a list of all registered model type strings."""
        return [m.value for m in ModelType]