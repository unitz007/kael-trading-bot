"""Time-aware train/validation/test splitting.

All splitters guarantee that training data never includes future
information relative to validation/test data — no data leakage.
"""

from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitResult:
    """Container for a single train/val/test split."""

    X_train: np.ndarray | pd.DataFrame
    X_val: np.ndarray | pd.DataFrame
    X_test: np.ndarray | pd.DataFrame
    y_train: np.ndarray | pd.Series
    y_val: np.ndarray | pd.Series
    y_test: np.ndarray | pd.Series


class TimeSeriesSplitter:
    """Time-aware data splitter.

    Given an index ordered chronologically, it splits data into
    train / validation / test sets where the train set always
    precedes the validation set, which in turn precedes the test set.

    Parameters
    ----------
    val_ratio:
        Proportion of data reserved for validation (between train and test).
    test_ratio:
        Proportion of data reserved for the final test (held-out) set.
    """

    def __init__(
        self,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> None:
        if not 0 < val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        if not 0 < test_ratio < 1:
            raise ValueError("test_ratio must be between 0 and 1")
        if val_ratio + test_ratio >= 1.0:
            raise ValueError(
                "val_ratio + test_ratio must be less than 1.0"
            )
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> SplitResult:
        """Perform a single chronological train/val/test split.

        Parameters
        ----------
        X:
            Feature matrix.
        y:
            Target variable.

        Returns
        -------
        SplitResult
        """
        n = len(X)
        test_start = int(n * (1.0 - self.test_ratio))
        val_start = int(test_start * (1.0 - self.val_ratio))

        return SplitResult(
            X_train=X[:val_start],
            X_val=X[val_start:test_start],
            X_test=X[test_start:],
            y_train=y[:val_start],
            y_val=y[val_start:test_start],
            y_test=y[test_start:],
        )

    def cross_validate_splits(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        n_splits: int = 5,
    ) -> Sequence[SplitResult]:
        """Generate *n_splits* chronological train/val splits.

        The test set is always the last ``test_ratio`` portion and is
        **not** included in the cross-validation splits.  Each fold
        expands the training set while keeping the validation set
        chronologically forward.

        Parameters
        ----------
        X:
            Feature matrix.
        y:
            Target variable.
        n_splits:
            Number of cross-validation folds.

        Returns
        -------
        list[SplitResult]
        """
        n = len(X)
        test_start = int(n * (1.0 - self.test_ratio))

        cv_X = X[:test_start]
        cv_y = y[:test_start]
        cv_n = test_start

        test_X = X[test_start:]
        test_y = y[test_start:]

        results: list[SplitResult] = []
        for i in range(1, n_splits + 1):
            train_end = int(cv_n * (1.0 - i / (n_splits + 1)))
            val_start_idx = train_end
            val_end_idx = int(cv_n * (1.0 - (i - 1) / (n_splits + 1)))

            results.append(
                SplitResult(
                    X_train=cv_X[:val_start_idx],
                    X_val=cv_X[val_start_idx:val_end_idx],
                    X_test=test_X,
                    y_train=cv_y[:val_start_idx],
                    y_val=cv_y[val_start_idx:val_end_idx],
                    y_test=test_y,
                )
            )

        return results