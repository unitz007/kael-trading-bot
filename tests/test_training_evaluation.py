"""Tests for model evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.kael_trading_bot.training.evaluation import (
    ClassificationMetrics,
    EvaluationResult,
    ModelEvaluator,
    TradingMetrics,
)


class TestClassificationMetrics:
    def test_to_dict(self) -> None:
        m = ClassificationMetrics(
            accuracy=0.9, precision=0.85, recall=0.8, f1=0.825, roc_auc=0.92
        )
        d = m.to_dict()
        assert d["accuracy"] == 0.9
        assert d["roc_auc"] == 0.92

    def test_to_dict_no_roc(self) -> None:
        m = ClassificationMetrics(
            accuracy=0.9, precision=0.85, recall=0.8, f1=0.825
        )
        d = m.to_dict()
        assert "roc_auc" not in d


class TestTradingMetrics:
    def test_to_dict(self) -> None:
        m = TradingMetrics(hit_rate=0.55, avg_return_per_trade=0.001)
        d = m.to_dict()
        assert d["hit_rate"] == 0.55
        assert "sharpe_ratio" not in d

    def test_to_dict_with_all(self) -> None:
        m = TradingMetrics(
            hit_rate=0.55,
            avg_return_per_trade=0.001,
            sharpe_ratio=1.2,
            max_drawdown=-0.05,
        )
        d = m.to_dict()
        assert "sharpe_ratio" in d
        assert "max_drawdown" in d


class TestModelEvaluator:
    def test_binary_classification(self) -> None:
        np.random.seed(42)
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.3, 0.8, 0.2, 0.4, 0.3, 0.7, 0.1, 0.8])

        evaluator = ModelEvaluator(average="binary")
        result = evaluator.evaluate(y_true, y_pred, y_proba)

        assert isinstance(result, EvaluationResult)
        assert 0.6 <= result.classification.accuracy <= 1.0
        assert result.classification.roc_auc is not None
        assert result.trading is None  # no returns provided

    def test_with_trading_metrics(self) -> None:
        np.random.seed(42)
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 1])
        returns = np.array([0.01, -0.02, 0.03, 0.04, -0.01, 0.02, -0.03, 0.01, 0.02, -0.01])

        evaluator = ModelEvaluator()
        result = evaluator.evaluate(y_true, y_pred, returns=returns)

        assert result.trading is not None
        assert 0.0 <= result.trading.hit_rate <= 1.0
        assert isinstance(result.trading.avg_return_per_trade, float)

    def test_no_proba(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])

        result = ModelEvaluator().evaluate(y_true, y_pred)
        assert result.classification.roc_auc is None

    def test_evaluation_result_to_dict(self) -> None:
        np.random.seed(42)
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])

        result = ModelEvaluator().evaluate(y_true, y_pred)
        d = result.to_dict()
        assert "classification" in d
        assert "accuracy" in d["classification"]
