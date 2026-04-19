"""Training run logger / observability.

Records training parameters, evaluation metrics, and model versions
to a JSON-lines file so that every training run is fully reproducible
and auditable.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TrainingLogger:
    """Structured logger for training runs.

    Each call to :meth:`log_run` appends a single JSON-lines record
    to the configured log file.  Records include:

    * timestamp
    * model_type, model_name, model_version
    * hyper-parameters
    * dataset sizes
    * evaluation metrics (train / val / test)
    * duration in seconds

    Parameters
    ----------
    log_file:
        Path to the JSON-lines log file.  Created (and parent
        directories made) on first write.
    """

    def __init__(self, log_file: str | Path = "logs/training_runs.jsonl") -> None:
        self.log_file = Path(log_file)

    def log_run(
        self,
        model_type: str,
        model_name: str,
        model_version: str,
        params: dict[str, Any],
        dataset_info: dict[str, Any],
        train_metrics: dict[str, Any] | None = None,
        val_metrics: dict[str, Any] | None = None,
        test_metrics: dict[str, Any] | None = None,
        duration_seconds: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a training-run record and return it.

        Parameters
        ----------
        model_type:
            Model type identifier (e.g. ``"xgboost"``).
        model_name:
            Descriptive name (e.g. ``"eurusd_xgboost"``).
        model_version:
            Version string.
        params:
            Hyper-parameters used for this run.
        dataset_info:
            Keys like ``n_train``, ``n_val``, ``n_test``, ``n_features``.
        train_metrics, val_metrics, test_metrics:
            Evaluation metric dicts for each split.
        duration_seconds:
            Wall-clock time of training in seconds.
        extra:
            Arbitrary additional fields.

        Returns
        -------
        dict
            The full record that was written.
        """
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_type": model_type,
            "model_name": model_name,
            "model_version": model_version,
            "params": params,
            "dataset": dataset_info,
        }
        if train_metrics is not None:
            record["train_metrics"] = train_metrics
        if val_metrics is not None:
            record["val_metrics"] = val_metrics
        if test_metrics is not None:
            record["test_metrics"] = test_metrics
        if duration_seconds is not None:
            record["duration_seconds"] = duration_seconds
        if extra is not None:
            record.update(extra)

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

        logger.info(
            "Logged training run: %s@%s (train_f1=%.4f)",
            model_name,
            model_version,
            (train_metrics or {}).get("f1", float("nan")),
        )
        return record

    def load_history(self) -> list[dict[str, Any]]:
        """Read all logged records from the log file.

        Returns an empty list if the file does not exist.
        """
        if not self.log_file.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.log_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
