"""Tests for model persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.kael_trading_bot.training.models import ModelRegistry
from src.kael_trading_bot.training.persistence import (
    ModelMetadata,
    ModelPersistence,
)


class TestModelMetadata:
    def test_round_trip(self) -> None:
        meta = ModelMetadata(
            model_type="xgboost",
            model_version="v1",
            params={"n_estimators": 100},
            metrics={"f1": 0.85},
            feature_names=["close", "rsi"],
        )
        d = meta.to_dict()
        restored = ModelMetadata.from_dict(d)
        assert restored == meta

    def test_defaults(self) -> None:
        meta = ModelMetadata(model_type="lr", model_version="v0")
        assert meta.params == {}
        assert meta.feature_names is None


class TestModelPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        pers = ModelPersistence(directory=str(tmp_path))
        model = ModelRegistry.create("logistic_regression")
        np.random.seed(42)
        X = np.random.randn(20, 3)
        y = np.random.randint(0, 2, 20)
        model.fit(X, y)

        meta = ModelMetadata(
            model_type="logistic_regression",
            model_version="v1",
            params={"C": 1.0},
        )

        saved_dir = pers.save(model, "test_model", "v1", meta)
        assert saved_dir.exists()

        loaded_model, loaded_meta = pers.load("test_model", "v1")
        assert loaded_meta.model_type == "logistic_regression"
        # Verify predictions match
        preds_orig = model.predict(X)
        preds_loaded = loaded_model.predict(X)
        np.testing.assert_array_equal(preds_orig, preds_loaded)

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        pers = ModelPersistence(directory=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            pers.load("nope", "v1")

    def test_list_versions(self, tmp_path: Path) -> None:
        pers = ModelPersistence(directory=str(tmp_path))
        model = ModelRegistry.create("logistic_regression")

        meta_v1 = ModelMetadata(model_type="lr", model_version="v1")
        meta_v2 = ModelMetadata(model_type="lr", model_version="v2")
        pers.save(model, "mymodel", "v1", meta_v1)
        pers.save(model, "mymodel", "v2", meta_v2)

        versions = pers.list_versions("mymodel")
        assert versions == ["v1", "v2"]

    def test_list_models(self, tmp_path: Path) -> None:
        pers = ModelPersistence(directory=str(tmp_path))
        model = ModelRegistry.create("logistic_regression")

        pers.save(model, "model_a", "v1", ModelMetadata(model_type="lr", model_version="v1"))
        pers.save(model, "model_b", "v1", ModelMetadata(model_type="lr", model_version="v1"))

        models = pers.list_models()
        assert "model_a" in models
        assert "model_b" in models

    def test_list_versions_empty(self, tmp_path: Path) -> None:
        pers = ModelPersistence(directory=str(tmp_path))
        assert pers.list_versions("nonexistent") == []

    def test_list_models_empty(self, tmp_path: Path) -> None:
        pers = ModelPersistence(directory=str(tmp_path))
        assert pers.list_models() == []
