"""Target variable generation for supervised learning.

Functions in this module create forward-looking label columns (targets)
from OHLCV data.  These targets represent the quantity the ML model will
learn to predict.

.. important::

    Targets are computed using **only current and past data** in the row
    (via ``shift``) so there is no look-ahead leakage.  However, the *last*
    ``horizon`` rows will contain NaN in the target column because the
    future is not yet known.  Callers should drop these rows before
    training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_future_return(
    df: pd.DataFrame,
    close_col: str = "close",
    horizon: int = 1,
    log: bool = True,
    prefix: str = "future_return",
) -> pd.DataFrame:
    """Append a future-return column.

    The future return at row *t* is:

    * **log**: ``log(close[t + horizon] / close[t])``
    * **simple**: ``(close[t + horizon] - close[t]) / close[t]``

    Parameters
    ----------
    df:
        OHLCV DataFrame with a ``close_col`` column.
    close_col:
        Name of the close-price column.
    horizon:
        Number of periods to look ahead.
    log:
        If ``True``, compute log-returns; otherwise simple returns.
    prefix:
        Column name prefix.  The full column name is
        ``<prefix>_<horizon>``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the target column added.  The last *horizon*
        rows will contain NaN.
    """
    result = df.copy()
    future_close = result[close_col].shift(-horizon)
    if log:
        ratio = future_close / result[close_col]
        result[f"{prefix}_{horizon}"] = np.where(ratio.notna() & (ratio > 0), np.log(ratio), pd.NA)
    else:
        result[f"{prefix}_{horizon}"] = (
            (future_close - result[close_col]) / result[close_col]
        )
    return result


def add_target_direction(
    df: pd.DataFrame,
    close_col: str = "close",
    horizon: int = 1,
    threshold: float = 0.0,
    prefix: str = "target_dir",
) -> pd.DataFrame:
    """Append a categorical direction target column.

    The value is:

    * ``1``  if future return > *threshold*  (up)
    * ``0``  if future return == *threshold*  (flat)
    * ``-1`` if future return < *threshold*   (down)

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    close_col:
        Name of the close-price column.
    horizon:
        Number of periods to look ahead.
    threshold:
        Minimum return magnitude to count as directional.  ``0.0``
        means any non-zero move counts.
    prefix:
        Column name prefix.  The full column name is
        ``<prefix>_<horizon>``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the direction column added.
    """
    result = df.copy()
    future_close = result[close_col].shift(-horizon)
    ret = (future_close - result[close_col]) / result[close_col]

    col = f"{prefix}_{horizon}"
    result[col] = 0
    result.loc[ret > threshold, col] = 1
    result.loc[ret < -threshold, col] = -1
    return result