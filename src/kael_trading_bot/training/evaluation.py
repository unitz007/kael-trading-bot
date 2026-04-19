"""Model evaluation metrics for classification and trading.

Provides both standard classification metrics and trading-oriented
metrics such as return-based scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Standard classification evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None = None

    def to_dict(self) -> dict[str, float]:
        d: dict[str, float] = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }
        if self.roc_auc is not None:
            d["roc_auc"] = self.roc_auc
        return d


@dataclass(frozen=True)
class TradingMetrics:
    """Trading-oriented evaluation metrics.

    These assume a long/short signal where the model predicts the
    direction of the next period's return.
    """

    hit_rate: float
    """Fraction of correct directional predictions."""

    avg_return_per_trade: float
    """Average return when following model signals."""

    sharpe_ratio: float | None = None
    """Sharpe-like ratio of strategy returns (annualised proxy)."""

    max_drawdown: float | None = None
    """Maximum observed drawdown of the strategy equity curve."""

    def to_dict(self) -> dict[str, float]:
        d: dict[str, float] = {
            "hit_rate": self.hit_rate,
            "avg_return_per_trade": self.avg_return_per_trade,
        }
        if self.sharpe_ratio is not None:
            d["sharpe_ratio"] = self.sharpe_ratio
        if self.max_drawdown is not None:
            d["max_drawdown"] = self.max_drawdown
        return d


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single model/dataset pair."""

    classification: ClassificationMetrics
    trading: TradingMetrics | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "classification": self.classification.to_dict(),
        }
        if self.trading is not None:
            d["trading"] = self.trading.to_dict()
        d.update(self.extra)
        return d


class ModelEvaluator:
    """Evaluate trained models on classification and trading metrics.

    Parameters
    ----------
    average:
        Averaging strategy for multi-class metrics
        (``'binary'`` | ``'macro'`` | ``'weighted'``).
    pos_label:
        Positive class label for binary classification.
    """

    def __init__(
        self,
        average: str = "binary",
        pos_label: int | str = 1,
    ) -> None:
        self.average = average
        self.pos_label = pos_label

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
        returns: np.ndarray | None = None,
    ) -> EvaluationResult:
        """Run full evaluation.

        Parameters
        ----------
        y_true:
            Ground-truth labels.
        y_pred:
            Predicted labels.
        y_proba:
            Predicted probabilities for the positive class (needed for
            ROC-AUC).
        returns:
            Array of per-period returns aligned with *y_true* / *y_pred*.
            When provided, trading-oriented metrics are computed.

        Returns
        -------
        EvaluationResult
        """
        clf = self._classification_metrics(y_true, y_pred, y_proba)
        trading = (
            self._trading_metrics(y_true, y_pred, returns) if returns is not None else None
        )
        return EvaluationResult(classification=clf, trading=trading)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None,
    ) -> ClassificationMetrics:
        roc_auc_val: float | None = None
        if y_proba is not None:
            try:
                roc_auc_val = float(
                    roc_auc_score(y_true, y_proba)
                )
            except ValueError:
                roc_auc_val = None

        return ClassificationMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, average=self.average, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, average=self.average, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, average=self.average, zero_division=0)),
            roc_auc=roc_auc_val,
        )

    @staticmethod
    def _trading_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        returns: np.ndarray,
    ) -> TradingMetrics:
        # Only count periods where the model makes an active signal
        mask = y_pred != 0 if np.any(y_pred == 0) else np.ones(len(y_pred), dtype=bool)
        correct = (y_pred[mask] == y_true[mask]).sum()
        total = mask.sum()
        hit_rate = correct / total if total > 0 else 0.0

        # Strategy returns: multiply direction signal by realised return
        direction = y_pred.astype(float)
        strategy_returns = direction * returns

        avg_ret = float(np.mean(strategy_returns[mask])) if total > 0 else 0.0

        sharpe: float | None = None
        if len(strategy_returns) > 1:
            std_ret = float(np.std(strategy_returns))
            sharpe = float(np.mean(strategy_returns) / std_ret) if std_ret > 0 else None

        max_dd: float | None = None
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        if len(drawdowns) > 0:
            max_dd = float(np.min(drawdowns))

        return TradingMetrics(
            hit_rate=hit_rate,
            avg_return_per_trade=avg_ret,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
        )