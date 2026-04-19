"""Training pipeline orchestrator.

Brings together model creation, data splitting, training, evaluation,
persistence, and logging into a single configurable pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.kael_trading_bot.training.evaluation import (
    EvaluationResult,
    ModelEvaluator,
)
from src.kael_trading_bot.training.logging import TrainingLogger
from src.kael_trading_bot.training.models import ModelRegistry, ModelType
from src.kael_trading_bot.training.persistence import (
    ModelMetadata,
    ModelPersistence,
)
from src.kael_trading_bot.training.splitting import TimeSeriesSplitter

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for a single training pipeline run.

    Attributes
    ----------
    model_type:
        Which model to train (see :class:`ModelType`).
    model_name:
        Human-readable name used for persistence and logging.
    model_version:
        Version string.  Defaults to a timestamp-based string.
    model_params:
        Hyper-parameter overrides merged on top of the defaults.
    val_ratio:
        Fraction of data for validation.
    test_ratio:
        Fraction of data for test.
    evaluator_average:
        Averaging strategy for classification metrics.
    save_model:
        Whether to persist the trained model to disk.
    log_run:
        Whether to log the training run via :class:`TrainingLogger`.
    cross_validate:
        If ``True``, perform time-series cross-validation *before*
        the final train on the full train+val split.
    n_cv_splits:
        Number of cross-validation folds (only used when
        ``cross_validate`` is ``True``).
    """

    model_type: ModelType | str = ModelType.XGBOOST
    model_name: str = "default_model"
    model_version: str = ""
    model_params: dict[str, Any] = field(default_factory=dict)
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    evaluator_average: str = "binary"
    save_model: bool = True
    log_run: bool = True
    cross_validate: bool = True
    n_cv_splits: int = 5

    def __post_init__(self) -> None:
        if isinstance(self.model_type, str):
            self.model_type = ModelType(self.model_type)
        if not self.model_version:
            self.model_version = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%S")


@dataclass
class PipelineResult:
    """Result of a completed training pipeline run."""

    model: Any
    model_name: str
    model_version: str
    model_type: str
    train_eval: EvaluationResult
    val_eval: EvaluationResult | None = None
    test_eval: EvaluationResult | None = None
    cv_results: list[dict[str, Any]] | None = None
    saved_path: str | None = None
    duration_seconds: float = 0.0

    @property
    def best_test_f1(self) -> float | None:
        if self.test_eval is not None:
            return self.test_eval.classification.f1
        return None


class TrainingPipeline:
    """End-to-end model training pipeline.

    Parameters
    ----------
    config:
        Pipeline configuration.
    evaluator:
        Custom :class:`ModelEvaluator` instance (optional).
    persistence:
        Custom :class:`ModelPersistence` instance (optional).
    training_logger:
        Custom :class:`TrainingLogger` instance (optional).
    """

    def __init__(
        self,
        config: PipelineConfig,
        evaluator: ModelEvaluator | None = None,
        persistence: ModelPersistence | None = None,
        training_logger: TrainingLogger | None = None,
    ) -> None:
        self.config = config
        self.evaluator = evaluator or ModelEvaluator(
            average=config.evaluator_average,
        )
        self.persistence = persistence or ModelPersistence()
        self.training_logger = training_logger or TrainingLogger()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        returns: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> PipelineResult:
        """Execute the full training pipeline.

        Steps:
        1. Time-aware train/val/test split.
        2. (Optional) Time-series cross-validation.
        3. Train on full train set.
        4. Evaluate on train, val, and test sets.
        5. Save model and log run.

        Parameters
        ----------
        X:
            Feature matrix (must be chronologically ordered).
        y:
            Target variable.
        returns:
            Per-period returns for trading-oriented metrics.
        feature_names:
            Optional list of feature column names.

        Returns
        -------
        PipelineResult
        """
        start_time = time.time()

        logger.info(
            "Starting pipeline: %s (%s) — %d samples, %d features",
            self.config.model_name,
            self.config.model_type.value,
            len(X),
            X.shape[1] if X.ndim == 2 else 1,
        )

        # 1. Split data
        splitter = TimeSeriesSplitter(
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
        )
        split = splitter.split(X, y)

        # 2. Optional cross-validation
        cv_results: list[dict[str, Any]] | None = None
        if self.config.cross_validate:
            cv_results = self._cross_validate(
                X, y, returns, splitter, feature_names
            )

        # 3. Train final model on train + val
        model = ModelRegistry.create(
            self.config.model_type,
            **self.config.model_params,
        )
        X_trainval = np.vstack(
            [
                np.asarray(split.X_train),
                np.asarray(split.X_val),
            ]
        )
        y_trainval = np.concatenate(
            [
                np.asarray(split.y_train),
                np.asarray(split.y_val),
            ]
        )

        model.fit(X_trainval, y_trainval)
        logger.info("Model training complete.")

        # 4. Evaluate
        train_eval = self._evaluate(model, split.X_train, split.y_train, returns[: len(split.y_train)] if returns is not None else None)
        val_eval = self._evaluate(model, split.X_val, split.y_val, returns[len(split.y_train): len(split.y_train) + len(split.y_val)] if returns is not None else None)
        test_eval = self._evaluate(model, split.X_test, split.y_test, returns[len(split.y_train) + len(split.y_val):] if returns is not None else None)

        duration = time.time() - start_time

        # 5. Save & log
        saved_path: str | None = None
        if self.config.save_model:
            meta = ModelMetadata(
                model_type=self.config.model_type.value,
                model_version=self.config.model_version,
                params={
                    **ModelRegistry.default_params(self.config.model_type),
                    **self.config.model_params,
                },
                metrics={
                    "train": train_eval.to_dict(),
                    "val": val_eval.to_dict() if val_eval else None,
                    "test": test_eval.to_dict() if test_eval else None,
                },
                feature_names=feature_names,
                trained_at=datetime.now(timezone.utc).isoformat(),
            )
            saved_path = str(
                self.persistence.save(
                    model,
                    self.config.model_name,
                    self.config.model_version,
                    meta,
                )
            )

        if self.config.log_run:
            self.training_logger.log_run(
                model_type=self.config.model_type.value,
                model_name=self.config.model_name,
                model_version=self.config.model_version,
                params={
                    **ModelRegistry.default_params(self.config.model_type),
                    **self.config.model_params,
                },
                dataset_info={
                    "n_total": len(X),
                    "n_train": len(split.y_train),
                    "n_val": len(split.y_val),
                    "n_test": len(split.y_test),
                    "n_features": X.shape[1] if X.ndim == 2 else 1,
                },
                train_metrics=train_eval.to_dict(),
                val_metrics=val_eval.to_dict() if val_eval else None,
                test_metrics=test_eval.to_dict() if test_eval else None,
                duration_seconds=duration,
            )

        logger.info(
            "Pipeline finished in %.1fs — test F1: %.4f",
            duration,
            test_eval.classification.f1 if test_eval else float("nan"),
        )

        return PipelineResult(
            model=model,
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            model_type=self.config.model_type.value,
            train_eval=train_eval,
            val_eval=val_eval,
            test_eval=test_eval,
            cv_results=cv_results,
            saved_path=saved_path,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        model: Any,
        X: Any,
        y: Any,
        returns: np.ndarray | None = None,
    ) -> EvaluationResult:
        y_pred = model.predict(np.asarray(X))
        try:
            y_proba = model.predict_proba(np.asarray(X))
            # For binary classification, keep probability of positive class
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
            else:
                y_proba = None
        except Exception:
            y_proba = None
        return self.evaluator.evaluate(
            np.asarray(y), y_pred, y_proba, returns
        )

    def _cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        returns: np.ndarray | None,
        splitter: TimeSeriesSplitter,
        feature_names: list[str] | None,
    ) -> list[dict[str, Any]]:
        splits = splitter.cross_validate_splits(
            X, y, n_splits=self.config.n_cv_splits
        )
        results: list[dict[str, Any]] = []
        for idx, split in enumerate(splits):
            model = ModelRegistry.create(
                self.config.model_type,
                **self.config.model_params,
            )
            model.fit(
                np.asarray(split.X_train),
                np.asarray(split.y_train),
            )
            eval_result = self._evaluate(
                model,
                split.X_val,
                split.y_val,
            )
            results.append(
                {
                    "fold": idx + 1,
                    "n_train": len(split.y_train),
                    "n_val": len(split.y_val),
                    "metrics": eval_result.to_dict(),
                }
            )
            logger.info(
                "CV fold %d/%d — val F1: %.4f",
                idx + 1,
                len(splits),
                eval_result.classification.f1,
            )
        return results