"""Tests for training run logger."""

from __future__ import annotations

from pathlib import Path

from src.kael_trading_bot.training.logging import TrainingLogger


class TestTrainingLogger:
    def test_log_and_load(self, tmp_path: Path) -> None:
        log_file = tmp_path / "runs.jsonl"
        logger = TrainingLogger(log_file=log_file)

        record = logger.log_run(
            model_type="xgboost",
            model_name="test_model",
            model_version="v1",
            params={"n_estimators": 100},
            dataset_info={"n_train": 800, "n_val": 100, "n_test": 100},
            train_metrics={"f1": 0.9},
            test_metrics={"f1": 0.82},
            duration_seconds=12.3,
        )

        assert record["model_type"] == "xgboost"
        assert record["test_metrics"]["f1"] == 0.82

        history = logger.load_history()
        assert len(history) == 1
        assert history[0]["model_name"] == "test_model"

    def test_load_empty(self, tmp_path: Path) -> None:
        log_file = tmp_path / "nonexistent.jsonl"
        logger = TrainingLogger(log_file=log_file)
        assert logger.load_history() == []

    def test_multiple_runs(self, tmp_path: Path) -> None:
        log_file = tmp_path / "runs.jsonl"
        logger = TrainingLogger(log_file=log_file)

        for i in range(5):
            logger.log_run(
                model_type="xgboost",
                model_name=f"model_{i}",
                model_version="v1",
                params={},
                dataset_info={},
            )

        history = logger.load_history()
        assert len(history) == 5

    def test_extra_fields(self, tmp_path: Path) -> None:
        log_file = tmp_path / "runs.jsonl"
        logger = TrainingLogger(log_file=log_file)

        record = logger.log_run(
            model_type="lr",
            model_name="m",
            model_version="v1",
            params={},
            dataset_info={},
            extra={"notes": "test run", "tag": "experimental"},
        )

        assert record["notes"] == "test run"
        assert record["tag"] == "experimental"

        history = logger.load_history()
        assert history[0]["tag"] == "experimental"
