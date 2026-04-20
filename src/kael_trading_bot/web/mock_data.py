"""Mock data for the web UI.

Provides realistic forex pair data that is used when the REST API
(issue #22) is not yet available.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

MOCK_PAIRS: list[dict[str, str | float]] = [
    {
        "symbol": "EUR/USD",
        "ticker": "EURUSD=X",
        "last_price": 1.0862,
        "change_pct": 0.15,
    },
    {
        "symbol": "GBP/USD",
        "ticker": "GBPUSD=X",
        "last_price": 1.2715,
        "change_pct": -0.08,
    },
    {
        "symbol": "USD/JPY",
        "ticker": "USDJPY=X",
        "last_price": 154.32,
        "change_pct": 0.42,
    },
    {
        "symbol": "AUD/USD",
        "ticker": "AUDUSD=X",
        "last_price": 0.6543,
        "change_pct": -0.21,
    },
    {
        "symbol": "USD/CAD",
        "ticker": "USDCAD=X",
        "last_price": 1.3621,
        "change_pct": 0.05,
    },
    {
        "symbol": "USD/CHF",
        "ticker": "USDCHF=X",
        "last_price": 0.8923,
        "change_pct": -0.12,
    },
]


def generate_mock_price_data(ticker: str, days: int = 90) -> list[dict[str, object]]:
    """Generate realistic OHLC price data for a given ticker.

    Args:
        ticker: The Yahoo Finance ticker symbol (e.g. ``EURUSD=X``).
        days: Number of historical days to generate.

    Returns:
        A list of dicts with keys: date, open, high, low, close, volume.
    """
    pair = next((p for p in MOCK_PAIRS if p["ticker"] == ticker), None)
    if pair is None:
        pair = {"symbol": ticker, "ticker": ticker, "last_price": 1.0, "change_pct": 0.0}

    base_price = float(pair["last_price"])
    volatility = base_price * 0.005  # 0.5% daily volatility

    random.seed(hash(ticker))  # Deterministic per ticker

    prices: list[dict[str, object]] = []
    current = base_price * (1 - volatility * days * 0.01)  # Start below current price

    end_date = datetime.now().date()

    for i in range(days, 0, -1):
        date = end_date - timedelta(days=i)
        change = random.gauss(0, volatility)
        open_price = current
        close_price = current + change
        high_price = max(open_price, close_price) + abs(random.gauss(0, volatility * 0.5))
        low_price = min(open_price, close_price) - abs(random.gauss(0, volatility * 0.5))

        prices.append({
            "date": date.isoformat(),
            "open": round(open_price, 4),
            "high": round(high_price, 4),
            "low": round(low_price, 4),
            "close": round(close_price, 4),
            "volume": random.randint(50_000, 500_000),
        })

        current = close_price

    return prices
