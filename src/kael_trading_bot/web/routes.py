"""Route definitions for the Kael Trading Bot web UI.

Provides pages for browsing forex pairs, viewing historical price data,
and placeholder pages for upcoming features.
"""

from __future__ import annotations

import os

import re
from flask import Blueprint, abort, render_template, request

from kael_trading_bot.web.mock_data import MOCK_PAIRS, generate_mock_price_data

main_bp = Blueprint("main", __name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("KAEL_API_BASE_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_json(path: str) -> dict | list | None:
    """Attempt to fetch JSON from the REST API.

    Returns ``None`` on any failure so callers can fall back to mock data.
    """
    try:
        import requests

        resp = requests.get(f"{API_BASE_URL}{path}", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@main_bp.route("/")
def index() -> str:
    """Redirect to the Forex Pairs listing page."""
    return render_template("pairs/index.html")


@main_bp.route("/pairs/")
def pairs_list() -> str:
    """Display all available forex pairs."""
    return render_template("pairs/index.html")


@main_bp.route("/pairs/<ticker>")
def pair_detail(ticker: str) -> str:
    """Display historical price data for a specific forex pair."""
    if not re.match(r"^[A-Za-z0-9=/^._-]+$", ticker):
        abort(400, description="Invalid ticker symbol")

    return render_template("pairs/detail.html", ticker=ticker)


@main_bp.route("/training/")
def training() -> str:
    """Placeholder page for Model Training."""
    return render_template(
        "placeholder.html",
        title="Model Training",
        description="Model training and management features are coming soon.",
    )


@main_bp.route("/predictions/")
def predictions() -> str:
    """Placeholder page for Predictions."""
    return render_template(
        "placeholder.html",
        title="Predictions",
        description="Prediction viewing and analysis features are coming soon.",
    )


# ---------------------------------------------------------------------------
# API endpoints (used by the frontend for data fetching)
# ---------------------------------------------------------------------------


@main_bp.route("/api/pairs")
def api_pairs() -> dict:
    """Return the list of available forex pairs.

    Tries the REST API first, falls back to mock data.
    """
    data = _fetch_json("/api/pairs")
    if data is not None:
        return {"pairs": data, "source": "api"}

    return {"pairs": MOCK_PAIRS, "source": "mock"}


@main_bp.route("/api/pairs/<ticker>/prices")
def api_pair_prices(ticker: str) -> dict:
    """Return historical price data for a given ticker.

    Tries the REST API first, falls back to mock data.
    """
    days = request.args.get("days", 90, type=int)
    days = max(1, min(days, 365))

    if not re.match(r"^[A-Za-z0-9=/^._-]+$", ticker):
        abort(400, description="Invalid ticker symbol")

    data = _fetch_json(f"/api/pairs/{ticker}/prices?days={days}")
    if data is not None:
        return {"prices": data, "source": "api"}

    return {
        "prices": generate_mock_price_data(ticker, days),
        "source": "mock",
    }