"""REST API routes for the web UI.

Provides JSON endpoints for:
- Listing available forex pairs
- Triggering model training
- Polling training job status
- Listing previously trained models
"""

from __future__ import annotations

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException

from kael_trading_bot.config import DEFAULT_FOREX_PAIRS
from kael_trading_bot.training.persistence import ModelPersistence
from kael_trading_bot.web.jobs import (
    TrainingJobStore,
    TrainingStatus,
    get_job_store,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["training"])

# Thread pool for running training jobs in background threads.
_executor = ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------------------
# Pairs
# ---------------------------------------------------------------------------


@router.get("/pairs")
async def list_pairs():
    """Return the list of configurable forex pairs."""
    return {
        "pairs": [
            {
                "ticker": p,
                "display": p.replace("=X", ""),
            }
            for p in DEFAULT_FOREX_PAIRS
        ]
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@router.post("/training/start")
async def start_training(
    pair: str,
    background_tasks: BackgroundTasks,
):
    """Start a training job for the given forex pair.

    Returns the created job object with status ``pending``.
    If a job is already running for that pair, returns 409 Conflict.
    """
    store = get_job_store()

    # Normalize pair: add =X suffix if missing
    ticker = pair if "=X" in pair else f"{pair}=X"

    # Check for existing running job for this pair
    existing = store.get_running_for_pair(ticker)
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "message": f"A training job is already running for {ticker}",
                "job": existing.to_dict(),
            },
        )

    job = store.create(ticker)
    job.status = TrainingStatus.RUNNING
    job.message = "Training started"

    background_tasks.add_task(_run_training, job.job_id, ticker)

    logger.info("Training job %s started for pair %s", job.job_id, ticker)
    return job.to_dict()


@router.get("/training/status")
async def list_training_status():
    """Return all training jobs with their current status."""
    store = get_job_store()
    jobs = store.list_all()
    return {"jobs": [j.to_dict() for j in jobs]}


@router.get("/training/status/{job_id}")
async def get_training_status(job_id: str):
    """Return the status of a specific training job."""
    store = get_job_store()
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@router.get("/models")
async def list_models():
    """Return all saved trained models with their metadata."""
    persistence = ModelPersistence()
    model_names = persistence.list_models()

    models = []
    for name in model_names:
        versions = persistence.list_versions(name)
        for version in versions:
            try:
                _, meta = persistence.load(name, version)
                models.append(
                    {
                        "name": name,
                        "version": version,
                        "model_type": meta.model_type,
                        "trained_at": meta.trained_at,
                        "metrics": meta.metrics.get("test", {}).get(
                            "classification", {}
                        )
                        if meta.metrics
                        else {},
                    }
                )
            except Exception:
                logger.warning(
                    "Failed to load metadata for %s/%s", name, version
                )

    return {"models": models}


# ---------------------------------------------------------------------------
# Background training worker
# ---------------------------------------------------------------------------


def _run_training(job_id: str, ticker: str) -> None:
    """Execute the full training pipeline in a background thread."""
    store = get_job_store()

    try:
        import numpy as np

        from kael_trading_bot.config import IngestionConfig
        from kael_trading_bot.features.pipeline import (
            FeatureConfig,
            build_feature_matrix,
        )
        from kael_trading_bot.ingestion import ForexDataFetcher
        from kael_trading_bot.training.pipeline import (
            PipelineConfig,
            TrainingPipeline,
        )

        store.update(job_id, message="Ingesting data…")

        # 1. Ingest
        ingestion_cfg = IngestionConfig(pairs=(ticker,))
        fetcher = ForexDataFetcher(ingestion_cfg)
        raw_df = fetcher.get(ticker)

        store.update(job_id, message="Engineering features…")

        # 2. Feature engineering
        raw_df.columns = [c.lower() for c in raw_df.columns]
        feature_df = build_feature_matrix(raw_df, config=FeatureConfig())

        # 3. Prepare arrays
        store.update(job_id, message="Preparing training data…")

        target_col = "target_direction_1"
        if target_col not in feature_df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in feature matrix."
            )

        exclude_cols = {target_col}
        for col in feature_df.columns:
            if col.startswith("future_return") or col.startswith("target_"):
                exclude_cols.add(col)

        feature_cols = [c for c in feature_df.columns if c not in exclude_cols]
        X = feature_df[feature_cols].values.astype(np.float64)
        y = feature_df[target_col].values.astype(np.float64)

        # 4. Train
        store.update(job_id, message="Training model…")

        model_name = ticker.replace("=", "_").replace("^", "_").lower()
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

        # Extract metrics
        metrics: dict[str, float] = {}
        if result.test_eval is not None:
            clf = result.test_eval.classification
            metrics = {
                "accuracy": round(clf.accuracy, 4),
                "precision": round(clf.precision, 4),
                "recall": round(clf.recall, 4),
                "f1": round(clf.f1, 4),
            }
            if clf.roc_auc is not None:
                metrics["roc_auc"] = round(clf.roc_auc, 4)

        store.update(
            job_id,
            status=TrainingStatus.COMPLETED,
            message="Training completed successfully",
            metrics=metrics,
            model_version=model_version,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            "Training job %s completed for %s — F1: %.4f",
            job_id,
            ticker,
            metrics.get("f1", 0),
        )

    except Exception as exc:
        store.update(
            job_id,
            status=TrainingStatus.FAILED,
            message=f"Training failed: {exc}",
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.error(
            "Training job %s failed for %s: %s\n%s",
            job_id,
            ticker,
            exc,
            traceback.format_exc(),
        )
