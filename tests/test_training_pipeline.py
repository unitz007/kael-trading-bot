"""Integration tests for the end-to-end training pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.kael_trading_bot.training.evaluation import ModelEvaluator
from src.kael_trading_bot.training.logging import TrainingLogger
from src.kael_trading_bot.training.models import ModelType
from src.kael_trading_bot.training.persistence import ModelPersistence
from src.kael_trading_bot.training.pipeline import PipelineConfig, PipelineResult, TrainingPipeline


def _make_dataset(n: int = 500, n_features: int = 10, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic binary classification dataset."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    # Simple linear separator so models can learn something
    y = (X[:, 0] + X[:, 1] + 0.5 * X[:, 2] > 0).astype(int)
    returns = rng.randn(n) * 0.02  # small random returns
    return X, y, returns


class TestPipelineConfig:
    def test_default_version_auto_generated(self) -> None:
        config = PipelineConfig(model_name="test")
        assert config.model_version  # non-empty

    def test_string_model_type_converted(self) -> None:
        config = PipelineConfig(model_type="logistic_regression")
        assert config.model_type is ModelType.LOGISTIC_REGRESSION


class TestTrainingPipeline:
    def test_decoder_accepts_score_matrix(self) -> None:
        label_values = np.array([-1.0, 0.0, 1.0])
        scores = np.array(
            [
                [0.1, 0.2, 0.7],
                [0.9, 0.05, 0.05],
                [0.0, 1.0, 0.0],
            ]
        )
        decoded = TrainingPipeline._decode_from_fit(scores, label_values)
        np.testing.assert_array_equal(decoded, np.array([1.0, -1.0, 0.0]))

    def test_decoder_truncates_extra_score_columns(self) -> None:
        label_values = np.array([-1.0, 1.0])
        scores = np.array(
            [
                [0.1, 0.9, 0.0],
                [0.8, 0.2, 0.0],
            ]
        )
        decoded = TrainingPipeline._decode_from_fit(scores, label_values)
        np.testing.assert_array_equal(decoded, np.array([1.0, -1.0]))

    def test_rejects_multilabel_indicator_targets(self) -> None:
        X, y, _ = _make_dataset(100)
        y_one_hot = np.eye(2, dtype=int)[y]
        config = PipelineConfig(
            model_type=ModelType.LOGISTIC_REGRESSION,
            model_name="reject_one_hot",
            model_version="v1",
            cross_validate=False,
            save_model=False,
            log_run=False,
        )
        pipeline = TrainingPipeline(config=config)
        with pytest.raises(ValueError, match="1D array of class labels"):
            pipeline.run(X, y_one_hot)

    def test_xgboost_accepts_direction_labels(self) -> None:
        """Regression: XGBoost rejects non-contiguous binary labels like [-1, 1]."""
        X, y01, _ = _make_dataset(300)
        y = np.where(y01 == 1, 1.0, -1.0)

        config = PipelineConfig(
            model_type=ModelType.XGBOOST,
            model_name="xgb_dir",
            model_version="v1",
            cross_validate=False,
            save_model=False,
            log_run=False,
            model_params={
                "n_estimators": 20,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "verbosity": 0,
            },
        )
        pipeline = TrainingPipeline(config=config)
        result = pipeline.run(X, y)

        assert result.test_eval is not None

    def test_logistic_regression_end_to_end(self, tmp_path: Path) -> None:
        X, y, returns = _make_dataset(300)
        config = PipelineConfig(
            model_type=ModelType.LOGISTIC_REGRESSION,
            model_name="test_lr",
            model_version="v1",
            cross_validate=False,
            save_model=True,
            log_run=True,
        )
        pipeline = TrainingPipeline(
            config=config,
            persistence=ModelPersistence(directory=str(tmp_path / "models")),
            training_logger=TrainingLogger(log_file=str(tmp_path / "logs" / "runs.jsonl")),
        )
        result = pipeline.run(X, y, returns=returns, feature_names=[f"f{i}" for i in range(X.shape[1])])

        assert isinstance(result, PipelineResult)
        assert result.model is not None
        assert result.train_eval is not None
        assert result.test_eval is not None
        assert result.saved_path is not None
        assert result.duration_seconds > 0
        assert 0.0 <= result.test_eval.classification.f1 <= 1.0
        assert result.test_eval.trading is not None

    def test_random_forest_with_cv(self, tmp_path: Path) -> None:
        X, y, returns = _make_dataset(400)
        config = PipelineConfig(
            model_type=ModelType.RANDOM_FOREST,
            model_name="test_rf",
            model_version="v1",
            cross_validate=True,
            n_cv_splits=3,
            save_model=False,
            log_run=False,
        )
        pipeline = TrainingPipeline(config=config)
        result = pipeline.run(X, y)

        assert result.cv_results is not None
        assert len(result.cv_results) == 3
        assert result.saved_path is None

    def test_model_persistence_round_trip(self, tmp_path: Path) -> None:
        X, y, _ = _make_dataset(200)
        config = PipelineConfig(
            model_type=ModelType.LOGISTIC_REGRESSION,
            model_name="persist_model",
            model_version="v2",
            cross_validate=False,
            save_model=True,
        )
        pers = ModelPersistence(directory=str(tmp_path / "models"))
        pipeline = TrainingPipeline(config=config, persistence=pers)
        result = pipeline.run(X, y)

        # Reload
        loaded_model, meta = pers.load("persist_model", "v2")
        preds_orig = result.model.predict(X)
        preds_loaded = loaded_model.predict(X)
        np.testing.assert_array_equal(preds_orig, preds_loaded)
        assert meta.model_type == "logistic_regression"

    def test_logging_records_run(self, tmp_path: Path) -> None:
        X, y, _ = _make_dataset(200)
        log_file = tmp_path / "logs.jsonl"
        tlog = TrainingLogger(log_file=log_file)
        config = PipelineConfig(
            model_type=ModelType.LOGISTIC_REGRESSION,
            model_name="log_test",
            model_version="v1",
            cross_validate=False,
            save_model=False,
            log_run=True,
        )
        pipeline = TrainingPipeline(config=config, training_logger=tlog)
        pipeline.run(X, y)

        history = tlog.load_history()
        assert len(history) == 1
        assert history[0]["model_name"] == "log_test"
        assert "train_metrics" in history[0]

    def test_custom_evaluator(self, tmp_path: Path) -> None:
        X, y, returns = _make_dataset(300)
        evaluator = ModelEvaluator(average="binary", pos_label=1)
        config = PipelineConfig(
            model_type=ModelType.LOGISTIC_REGRESSION,
            model_name="custom_eval",
            cross_validate=False,
            save_model=False,
        )
        pipeline = TrainingPipeline(config=config, evaluator=evaluator)
        result = pipeline.run(X, y, returns=returns)
        assert result.test_eval is not None

    def test_best_test_f1_property(self, tmp_path: Path) -> None:
        X, y, _ = _make_dataset(200)
        config = PipelineConfig(
            model_type=ModelType.LOGISTIC_REGRESSION,
            model_name="f1_test",
            cross_validate=False,
            save_model=False,
            log_run=False,
        )
        pipeline = TrainingPipeline(config=config)
        result = pipeline.run(X, y)
        assert result.best_test_f1 is not None
        assert 0.0 <= result.best_test_f1 <= 1.0

    def test_no_test_eval_when_test_empty(self) -> None:
        """Edge case: very small dataset where test split might be tiny."""
        X, y, _ = _make_dataset(20)
        config = PipelineConfig(
            model_type=ModelType.LOGISTIC_REGRESSION,
            model_name="small_test",
            cross_validate=False,
            save_model=False,
            log_run=False,
        )
        pipeline = TrainingPipeline(config=config)
        result = pipeline.run(X, y)
        # Pipeline should not crash; test_eval may have degenerate metrics
        assert result.test_eval is not None
