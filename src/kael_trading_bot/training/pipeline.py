"""Training pipeline orchestrator (fixed multiclass + time-series safe version)."""

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


# =========================================================
# CONFIG
# =========================================================

@dataclass
class PipelineConfig:
    model_type: ModelType | str = ModelType.XGBOOST
    model_name: str = "default_model"
    model_version: str = ""
    model_params: dict[str, Any] = field(default_factory=dict)

    val_ratio: float = 0.15
    test_ratio: float = 0.15

    evaluator_average: str = "macro"

    save_model: bool = True
    log_run: bool = True

    cross_validate: bool = True
    n_cv_splits: int = 5

    def __post_init__(self) -> None:
        if isinstance(self.model_type, str):
            self.model_type = ModelType(self.model_type)

        if not self.model_version:
            self.model_version = datetime.now(timezone.utc).strftime(
                "v%Y%m%dT%H%M%S"
            )


# =========================================================
# RESULT
# =========================================================

@dataclass
class PipelineResult:
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
        return self.test_eval.classification.f1 if self.test_eval else None


# =========================================================
# PIPELINE
# =========================================================

class TrainingPipeline:

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

    # =====================================================
    # LABEL HANDLING (PER-FIT)
    # =====================================================

    @staticmethod
    def _normalize_y(y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y)
        if y_arr.ndim == 2:
            # Accept column vectors (n, 1) but reject true multi-output /
            # multilabel indicator targets (n, k) where k > 1.
            if y_arr.shape[1] == 1:
                y_arr = y_arr.reshape(-1)
            else:
                raise ValueError(
                    "y must be a 1D array of class labels. "
                    f"Got shape={y_arr.shape} (looks like multi-output / one-hot labels)."
                )
        elif y_arr.ndim != 1:
            raise ValueError(f"y must be 1D; got shape={y_arr.shape}.")
        if y_arr.size > 0 and not np.all(np.isfinite(y_arr)):
            raise ValueError("y contains NaN/inf; labels must be finite.")
        return y_arr

    @staticmethod
    def _encode_for_fit(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode labels as contiguous indices [0..k-1] for fitting."""
        y_arr = TrainingPipeline._normalize_y(y)
        label_values, y_enc = np.unique(y_arr, return_inverse=True)
        return label_values, y_enc.astype(int, copy=False)

    @staticmethod
    def _decode_from_fit(y_pred: np.ndarray, label_values: np.ndarray) -> np.ndarray:
        """Decode predicted indices back to original label values."""
        y_arr = np.asarray(y_pred)
        if y_arr.size == 0:
            return y_arr
        if y_arr.ndim == 2 and y_arr.shape[1] == 1:
            y_arr = y_arr.reshape(-1)
        elif y_arr.ndim == 2 and y_arr.shape[1] >= int(label_values.size) and int(label_values.size) > 0:
            # Some estimators return per-class scores/probabilities from predict().
            # Convert to class indices.
            if y_arr.shape[1] != int(label_values.size):
                logger.warning(
                    "Model predict returned %d columns but training saw %d classes; "
                    "using the first %d columns for argmax decoding.",
                    y_arr.shape[1],
                    int(label_values.size),
                    int(label_values.size),
                )
                y_arr = y_arr[:, : int(label_values.size)]
            y_arr = np.argmax(y_arr, axis=1).astype(int, copy=False)
        elif y_arr.ndim != 1:
            raise ValueError(
                "Model predict returned an unsupported output shape. "
                f"Got shape={y_arr.shape}, expected (n,) or (n, {int(label_values.size)})."
            )
        if not np.issubdtype(y_arr.dtype, np.integer):
            y_arr_int = np.rint(y_arr).astype(int)
            if not np.allclose(y_arr, y_arr_int):
                return y_arr
            y_arr = y_arr_int
        return label_values[y_arr]

    # =====================================================
    # SAFE TRAINING HELPERS
    # =====================================================

    def _safe_fit(self, model, X, y):
        """Avoid crashes when a fold misses classes."""
        label_values, y_enc = self._encode_for_fit(y)
        if label_values.size < 2:
            return None
        model.fit(X, y_enc)
        return model, label_values

    # =====================================================
    # PUBLIC RUN
    # =====================================================

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        returns: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> PipelineResult:

        start_time = time.time()

        y = self._normalize_y(y)
        if len(y) != len(X):
            raise ValueError(f"X and y must have the same number of samples; got len(X)={len(X)} len(y)={len(y)}.")

        logger.info(
            "Starting pipeline %s (%s) — samples=%d features=%d",
            self.config.model_name,
            self.config.model_type.value,
            len(X),
            X.shape[1] if X.ndim == 2 else 1,
        )

        # -------------------------------------------------
        # SPLIT
        # -------------------------------------------------
        splitter = TimeSeriesSplitter(
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
        )
        split = splitter.split(X, y)

        # -------------------------------------------------
        # CROSS VALIDATION (SAFE)
        # -------------------------------------------------
        cv_results = []
        if self.config.cross_validate:
            cv_results = self._cross_validate(X, y, splitter)

        # -------------------------------------------------
        # TRAIN FINAL MODEL
        # -------------------------------------------------
        model = ModelRegistry.create(
            self.config.model_type,
            **self.config.model_params,
        )

        X_trainval = np.vstack([split.X_train, split.X_val])
        y_trainval = np.concatenate([split.y_train, split.y_val])

        fit_label_values, y_trainval_enc = self._encode_for_fit(y_trainval)
        if fit_label_values.size < 2:
            raise ValueError("Training labels contain only one class; cannot fit a classifier.")
        model.fit(X_trainval, y_trainval_enc)

        logger.info("Model training complete.")

        # -------------------------------------------------
        # EVALUATION
        # -------------------------------------------------
        train_eval = self._evaluate(
            model,
            split.X_train,
            split.y_train,
            returns[: len(split.y_train)] if returns is not None else None,
            label_values=fit_label_values,
        )

        val_eval = self._evaluate(
            model,
            split.X_val,
            split.y_val,
            returns[len(split.y_train): len(split.y_train) + len(split.y_val)]
            if returns is not None else None,
            label_values=fit_label_values,
        )

        test_eval = self._evaluate(
            model,
            split.X_test,
            split.y_test,
            returns[len(split.y_train) + len(split.y_val):]
            if returns is not None else None,
            label_values=fit_label_values,
        )

        duration = time.time() - start_time

        # -------------------------------------------------
        # SAVE MODEL
        # -------------------------------------------------
        saved_path = None
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

        # -------------------------------------------------
        # LOG RUN
        # -------------------------------------------------
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
                },
                train_metrics=train_eval.to_dict(),
                val_metrics=val_eval.to_dict() if val_eval else None,
                test_metrics=test_eval.to_dict() if test_eval else None,
                duration_seconds=duration,
            )

        logger.info(
            "Finished in %.1fs — test F1: %.4f",
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

    # =====================================================
    # EVALUATION
    # =====================================================

    def _evaluate(self, model, X, y, returns=None, *, label_values: np.ndarray | None = None):

        y_pred = model.predict(np.asarray(X))

        y_true = self._normalize_y(np.asarray(y))
        if label_values is not None:
            y_pred = self._decode_from_fit(y_pred, label_values)

        try:
            y_proba = model.predict_proba(np.asarray(X))
            if y_proba.ndim == 2 and label_values is not None and y_proba.shape[1] > int(label_values.size):
                logger.warning(
                    "Model predict_proba returned %d columns but training saw %d classes; "
                    "using the first %d columns.",
                    y_proba.shape[1],
                    int(label_values.size),
                    int(label_values.size),
                )
                y_proba = y_proba[:, : int(label_values.size)]
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                pos_col = 1
                if label_values is not None:
                    matches = np.where(label_values == self.evaluator.pos_label)[0]
                    if matches.size == 1:
                        pos_col = int(matches[0])
                y_proba = y_proba[:, pos_col]
        except Exception:
            y_proba = None

        return self.evaluator.evaluate(y_true, y_pred, y_proba, returns)

    # =====================================================
    # SAFE CROSS VALIDATION
    # =====================================================

    def _cross_validate(self, X, y, splitter):

        splits = splitter.cross_validate_splits(
            X, y,
            n_splits=self.config.n_cv_splits
        )

        results = []

        for idx, split in enumerate(splits):

            model = ModelRegistry.create(
                self.config.model_type,
                **self.config.model_params,
            )

            # skip invalid folds (key fix)
            fold_label_values, y_train_enc = self._encode_for_fit(split.y_train)
            if fold_label_values.size < 2:
                logger.warning("Skipping fold %d (insufficient classes)", idx + 1)
                continue

            model.fit(split.X_train, y_train_enc)

            eval_result = self._evaluate(
                model,
                split.X_val,
                split.y_val,
                label_values=fold_label_values,
            )

            results.append({
                "fold": idx + 1,
                "metrics": eval_result.to_dict(),
            })

            logger.info(
                "CV fold %d — F1: %.4f",
                idx + 1,
                eval_result.classification.f1,
            )

        return results
