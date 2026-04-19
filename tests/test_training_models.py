"""Tests for the model registry and factory."""

from __future__ import annotations

import pytest

from src.kael_trading_bot.training.models import ModelRegistry, ModelType


class TestModelType:
    def test_values(self) -> None:
        assert ModelType.XGBOOST.value == "xgboost"
        assert ModelType.LIGHTGBM.value == "lightgbm"
        assert ModelType.LOGISTIC_REGRESSION.value == "logistic_regression"
        assert ModelType.RANDOM_FOREST.value == "random_forest"

    def test_from_string(self) -> None:
        assert ModelType("xgboost") is ModelType.XGBOOST

    def test_invalid_string(self) -> None:
        with pytest.raises(ValueError):
            ModelType("nonexistent")


class TestModelRegistry:
    def test_available_models(self) -> None:
        models = ModelRegistry.available_models()
        assert "xgboost" in models
        assert "lightgbm" in models
        assert "logistic_regression" in models
        assert "random_forest" in models

    def test_default_params_returns_copy(self) -> None:
        p1 = ModelRegistry.default_params(ModelType.XGBOOST)
        p2 = ModelRegistry.default_params(ModelType.XGBOOST)
        assert p1 is not p2
        assert p1 == p2

    def test_create_logistic_regression(self) -> None:
        model = ModelRegistry.create(ModelType.LOGISTIC_REGRESSION)
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_create_random_forest(self) -> None:
        model = ModelRegistry.create(ModelType.RANDOM_FOREST)
        assert model is not None
        assert hasattr(model, "fit")

    def test_create_with_overrides(self) -> None:
        model = ModelRegistry.create(
            ModelType.LOGISTIC_REGRESSION, C=0.5, max_iter=500
        )
        assert model.C == 0.5  # type: ignore[attr-defined]
        assert model.max_iter == 500  # type: ignore[attr-defined]

    def test_create_unsupported_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported model type"):
            ModelRegistry.create("nonexistent")

    def test_create_from_string(self) -> None:
        model = ModelRegistry.create("logistic_regression")
        assert model is not None
