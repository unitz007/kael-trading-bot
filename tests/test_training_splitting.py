"""Tests for time-aware data splitting."""

from __future__ import annotations

import numpy as np
import pytest

from src.kael_trading_bot.training.splitting import (
    SplitResult,
    TimeSeriesSplitter,
)


class TestTimeSeriesSplitterInit:
    def test_valid_ratios(self) -> None:
        splitter = TimeSeriesSplitter(val_ratio=0.1, test_ratio=0.1)
        assert splitter.val_ratio == 0.1
        assert splitter.test_ratio == 0.1

    def test_invalid_val_ratio(self) -> None:
        with pytest.raises(ValueError, match="val_ratio"):
            TimeSeriesSplitter(val_ratio=0.0)

    def test_invalid_test_ratio(self) -> None:
        with pytest.raises(ValueError, match="test_ratio"):
            TimeSeriesSplitter(test_ratio=1.5)

    def test_sum_too_large(self) -> None:
        with pytest.raises(ValueError, match="val_ratio \\+ test_ratio"):
            TimeSeriesSplitter(val_ratio=0.5, test_ratio=0.6)


class TestSplit:
    def test_split_shapes(self) -> None:
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        splitter = TimeSeriesSplitter(val_ratio=0.2, test_ratio=0.2)
        result = splitter.split(X, y)

        assert isinstance(result, SplitResult)
        total = len(result.y_train) + len(result.y_val) + len(result.y_test)
        assert total == 100

    def test_no_overlap(self) -> None:
        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = np.arange(200)  # unique index to detect overlap

        splitter = TimeSeriesSplitter(val_ratio=0.15, test_ratio=0.15)
        result = splitter.split(X, y)

        train_set = set(result.y_train.tolist())
        val_set = set(result.y_val.tolist())
        test_set = set(result.y_test.tolist())

        assert len(train_set & val_set) == 0
        assert len(val_set & test_set) == 0
        assert len(train_set & test_set) == 0

    def test_chronological_order(self) -> None:
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = np.arange(100)

        result = TimeSeriesSplitter(val_ratio=0.1, test_ratio=0.1).split(X, y)

        # All train indices < all val indices < all test indices
        assert result.y_train.max() < result.y_val.min()
        assert result.y_val.max() < result.y_test.min()


class TestCrossValidate:
    def test_number_of_splits(self) -> None:
        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = np.random.randint(0, 2, 200)

        results = TimeSeriesSplitter(val_ratio=0.15, test_ratio=0.15).cross_validate_splits(
            X, y, n_splits=3
        )
        assert len(results) == 3

    def test_no_future_leakage(self) -> None:
        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = np.arange(200)

        results = TimeSeriesSplitter(val_ratio=0.15, test_ratio=0.15).cross_validate_splits(
            X, y, n_splits=5
        )
        for split in results:
            assert split.y_train.max() < split.y_val.min()

    def test_test_set_is_consistent(self) -> None:
        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = np.arange(200)

        results = TimeSeriesSplitter(val_ratio=0.15, test_ratio=0.15).cross_validate_splits(
            X, y, n_splits=3
        )
        # All splits should share the same test set
        test_sets = [set(s.y_test.tolist()) for s in results]
        for ts in test_sets[1:]:
            assert ts == test_sets[0]
