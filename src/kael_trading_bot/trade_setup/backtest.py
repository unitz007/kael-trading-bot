"""Backtest-driven R:R ratio optimisation.

Analyses historical OHLCV data to determine which risk-to-reward ratio
produced the best historical win-rate × R:R score for a given currency pair
and timeframe.  The selected ratio is clamped to a configurable floor
(default 1:1.2) to guarantee a minimum quality threshold.

Strategy
--------
1. Compute a rolling ATR(14) for volatility.
2. For each candidate R:R ratio (1:1.2 … 1:3.0 in 0.1 steps):
   a. Walk through the historical data simulating entry at every *N*-bar
      interval (where *N* is derived from ``lookback_window``).
   b. Entry direction is determined by a simple momentum signal: price
      change over ``signal_period`` bars vs an ATR-normalised threshold.
   c. Stop-loss = entry ± ATR × sl_mult  (always 1× ATR).
   d. Take-profit = entry ± ATR × (sl_mult × rr_ratio).
   e. A trade hits TP or SL — whichever comes first based on the future
      high/low bars up to ``max_hold_period`` bars ahead.
   f. Record win/loss.
3. Score = win_rate × rr_ratio  (rewards higher win-rate AND higher reward).
4. Return the rr_ratio with the highest score, clamped to ``min_rr``.
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

MIN_RR_RATIO: float = 1.2
"""Floor for the R:R ratio.  Any backtest-derived ratio below this is clamped."""

MAX_RR_RATIO: float = 3.0
"""Ceiling for candidate R:R ratios."""

RR_STEP: float = 0.1
"""Step size when enumerating candidate R:R ratios."""

SIGNAL_PERIOD: int = 5
"""Number of bars used to determine entry direction (momentum signal)."""

SIGNAL_ATR_MULT: float = 0.3
"""Minimum ATR-multiple move to trigger a directional entry."""

LOOKBACK_WINDOW: int = 200
"""Number of recent bars to include in the backtest (most recent data is
weighted more)."""

ENTRY_SPACING: int = 5
"""Minimum bars between consecutive simulated entries."""

MAX_HOLD_PERIOD: int = 30
"""Maximum bars a simulated trade can be held before it is force-closed."""

SL_ATR_MULT: float = 1.0
"""Stop-loss distance as a multiple of ATR (fixed at 1×)."""


# ---------------------------------------------------------------------------
# ATR computation
# ---------------------------------------------------------------------------

def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range from a DataFrame with OHLC columns."""
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr


# ---------------------------------------------------------------------------
# Core backtest for a single R:R candidate
# ---------------------------------------------------------------------------

def _backtest_rr(
    df: pd.DataFrame,
    atr: pd.Series,
    rr_ratio: float,
    *,
    signal_period: int = SIGNAL_PERIOD,
    signal_atr_mult: float = SIGNAL_ATR_MULT,
    entry_spacing: int = ENTRY_SPACING,
    max_hold: int = MAX_HOLD_PERIOD,
    sl_mult: float = SL_ATR_MULT,
) -> tuple[int, int, int]:
    """Run a simple backtest for a given R:R ratio.

    Returns
    -------
    (wins, losses, total_trades)
    """
    close = df["Close"].astype(float).values
    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values
    atr_vals = atr.values

    wins = 0
    losses = 0
    total = 0

    i = signal_period  # need enough history for the signal
    while i < len(df) - max_hold:
        # --- entry signal ---
        price_change = close[i] - close[i - signal_period]
        atr_now = atr_vals[i]

        if np.isnan(atr_now) or atr_now <= 0:
            i += 1
            continue

        # Determine direction
        if price_change > atr_now * signal_atr_mult:
            direction = 1  # buy
        elif price_change < -atr_now * signal_atr_mult:
            direction = -1  # sell
        else:
            i += 1
            continue  # no signal

        entry = close[i]
        sl_distance = atr_now * sl_mult
        tp_distance = atr_now * sl_mult * rr_ratio

        if direction == 1:
            sl_price = entry - sl_distance
            tp_price = entry + tp_distance
        else:
            sl_price = entry + sl_distance
            tp_price = entry - tp_distance

        # Walk forward to see if SL or TP is hit first
        hit = None
        for j in range(i + 1, min(i + max_hold + 1, len(df))):
            if direction == 1:
                if low[j] <= sl_price:
                    hit = "sl"
                    break
                if high[j] >= tp_price:
                    hit = "tp"
                    break
            else:
                if high[j] >= sl_price:
                    hit = "sl"
                    break
                if low[j] <= tp_price:
                    hit = "tp"
                    break

        if hit == "tp":
            wins += 1
            total += 1
        elif hit == "sl":
            losses += 1
            total += 1
        # else trade never resolved — skip

        i += entry_spacing

    return wins, losses, total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_optimal_rr_ratio(
    ohlcv_df: pd.DataFrame,
    *,
    min_rr: float = MIN_RR_RATIO,
    max_rr: float = MAX_RR_RATIO,
    step: float = RR_STEP,
    lookback_window: int = LOOKBACK_WINDOW,
) -> dict:
    """Select the optimal R:R ratio from backtested historical data.

    Parameters
    ----------
    ohlcv_df:
        DataFrame with at least ``High``, ``Low``, ``Close`` columns and a
        datetime index.
    min_rr:
        Minimum allowed R:R ratio (floor).
    max_rr:
        Maximum candidate R:R ratio (ceiling).
    step:
        Step size for enumerating candidate ratios.
    lookback_window:
        Number of most-recent bars to include in the backtest.

    Returns
    -------
    dict with keys:
        - ``rr_ratio``  (float): The chosen R:R ratio.
        - ``min_rr``    (float): The configured floor.
        - ``backtested`` (bool): Whether real backtest data was used.
        - ``candidates`` (list[dict]): All candidate ratios with scores.
        - ``reason``    (str): Human-readable explanation.
    """
    if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < lookback_window // 2:
        logger.warning(
            "Insufficient OHLCV data (%d rows) for R:R backtest; "
            "using minimum ratio %.2f",
            len(ohlcv_df) if ohlcv_df is not None else 0,
            min_rr,
        )
        return {
            "rr_ratio": min_rr,
            "min_rr": min_rr,
            "backtested": False,
            "candidates": [],
            "reason": (
                "Insufficient historical data for backtesting; "
                f"using minimum R:R ratio of 1:{min_rr:.2f}"
            ),
        }

    # Work with the most recent ``lookback_window`` rows
    df = ohlcv_df.iloc[-lookback_window:].copy()

    # Compute ATR
    atr = _compute_atr(df)

    # Enumerate candidate ratios
    candidates: list[float] = []
    r = min_rr
    while r <= max_rr + 1e-9:
        candidates.append(round(r, 2))
        r += step

    best_rr = min_rr
    best_score = -1.0
    results: list[dict] = []

    for rr in candidates:
        wins, losses, total = _backtest_rr(df, atr, rr)
        win_rate = wins / total if total > 0 else 0.0
        # Score: win_rate × rr — rewards both high win-rate and high reward
        score = win_rate * rr if total > 0 else 0.0

        results.append({
            "rr_ratio": rr,
            "wins": wins,
            "losses": losses,
            "total_trades": total,
            "win_rate": round(win_rate, 4),
            "score": round(score, 4),
        })

        if score > best_score:
            best_score = score
            best_rr = rr

    # Clamp to minimum
    if best_rr < min_rr:
        best_rr = min_rr
        reason = (
            f"Backtest best ratio 1:{best_rr:.2f} was below minimum; "
            f"clamped to 1:{min_rr:.2f}"
        )
    else:
        reason = (
            f"Selected 1:{best_rr:.2f} based on backtest of "
            f"{len(df)} historical bars (score={best_score:.4f})"
        )

    logger.info(
        "Dynamic R:R selection: chose 1:%.2f (score=%.4f) from %d candidates "
        "over %d bars. %s",
        best_rr,
        best_score,
        len(candidates),
        len(df),
        reason,
    )

    return {
        "rr_ratio": round(best_rr, 2),
        "min_rr": min_rr,
        "backtested": True,
        "candidates": results,
        "reason": reason,
    }
