"""Tests for the Kael Trading Bot web application."""

from __future__ import annotations

import json

import pytest

from kael_trading_bot.web import create_app
from kael_trading_bot.web.mock_data import MOCK_PAIRS, generate_mock_price_data


@pytest.fixture()
def app():
    """Create a test Flask application."""
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture()
def client(app):
    """Create a test client."""
    return app.test_client()


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------


class TestPageRoutes:
    """Tests for HTML page routes."""

    def test_index_redirects_to_pairs(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_pairs_list_page(self, client):
        resp = client.get("/pairs/")
        assert resp.status_code == 200
        assert b"Forex Pairs" in resp.data

    def test_pair_detail_page(self, client):
        ticker = MOCK_PAIRS[0]["ticker"]
        resp = client.get(f"/pairs/{ticker}")
        assert resp.status_code == 200
        assert ticker.encode() in resp.data

    def test_pair_detail_invalid_ticker(self, client):
        resp = client.get("/pairs/<script>")
        assert resp.status_code == 400

    def test_training_placeholder(self, client):
        resp = client.get("/training/")
        assert resp.status_code == 200
        assert b"Model Training" in resp.data
        assert b"coming soon" in resp.data

    def test_predictions_placeholder(self, client):
        resp = client.get("/predictions/")
        assert resp.status_code == 200
        assert b"Predictions" in resp.data
        assert b"coming soon" in resp.data


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


class TestApiRoutes:
    """Tests for JSON API routes."""

    def test_api_pairs_returns_list(self, client):
        resp = client.get("/api/pairs")
        assert resp.status_code == 200

        data = json.loads(resp.data)
        assert "pairs" in data
        assert "source" in data
        assert len(data["pairs"]) > 0

    def test_api_pair_prices_returns_data(self, client):
        ticker = MOCK_PAIRS[0]["ticker"]
        resp = client.get(f"/api/pairs/{ticker}/prices")
        assert resp.status_code == 200

        data = json.loads(resp.data)
        assert "prices" in data
        assert "source" in data
        assert len(data["prices"]) > 0

        price = data["prices"][0]
        assert "date" in price
        assert "open" in price
        assert "high" in price
        assert "low" in price
        assert "close" in price
        assert "volume" in price

    def test_api_pair_prices_days_param(self, client):
        ticker = MOCK_PAIRS[0]["ticker"]
        resp = client.get(f"/api/pairs/{ticker}/prices?days=30")
        assert resp.status_code == 200

        data = json.loads(resp.data)
        assert len(data["prices"]) == 30

    def test_api_pair_prices_clamps_days(self, client):
        """Days should be clamped to [1, 365]."""
        ticker = MOCK_PAIRS[0]["ticker"]

        resp_large = client.get(f"/api/pairs/{ticker}/prices?days=999")
        data = json.loads(resp_large.data)
        assert len(data["prices"]) == 365

        resp_zero = client.get(f"/api/pairs/{ticker}/prices?days=0")
        data = json.loads(resp_zero.data)
        assert len(data["prices"]) == 1

    def test_api_pair_prices_invalid_ticker(self, client):
        resp = client.get("/api/pairs/<script>/prices")
        assert resp.status_code == 400

    def test_api_pair_prices_unknown_ticker(self, client):
        resp = client.get("/api/pairs/FOOBAR/prices")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert len(data["prices"]) > 0  # Mock data is generated


# ---------------------------------------------------------------------------
# Navigation bar
# ---------------------------------------------------------------------------


class TestNavigation:
    """Tests for shared layout navigation links."""

    def test_navbar_links_present_on_pairs_page(self, client):
        resp = client.get("/pairs/")
        assert resp.status_code == 200
        html = resp.data.decode()

        assert 'href="/pairs/"' in html
        assert 'href="/training/"' in html
        assert 'href="/predictions/"' in html

    def test_navbar_links_present_on_detail_page(self, client):
        ticker = MOCK_PAIRS[0]["ticker"]
        resp = client.get(f"/pairs/{ticker}")
        assert resp.status_code == 200
        html = resp.data.decode()

        assert 'href="/training/"' in html
        assert 'href="/predictions/"' in html

    def test_back_link_on_detail_page(self, client):
        ticker = MOCK_PAIRS[0]["ticker"]
        resp = client.get(f"/pairs/{ticker}")
        assert resp.status_code == 200
        html = resp.data.decode()

        assert 'href="/pairs/"' in html


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------


class TestMockData:
    """Tests for mock data generation."""

    def test_mock_pairs_have_required_fields(self):
        for pair in MOCK_PAIRS:
            assert "symbol" in pair
            assert "ticker" in pair
            assert "last_price" in pair
            assert "change_pct" in pair

    def test_generate_mock_price_data_returns_correct_count(self):
        data = generate_mock_price_data("EURUSD=X", days=30)
        assert len(data) == 30

    def test_generate_mock_price_data_has_required_fields(self):
        data = generate_mock_price_data("EURUSD=X", days=1)
        row = data[0]
        assert "date" in row
        assert "open" in row
        assert "high" in row
        assert "low" in row
        assert "close" in row
        assert "volume" in row

    def test_generate_mock_price_data_is_deterministic(self):
        data1 = generate_mock_price_data("EURUSD=X", days=10)
        data2 = generate_mock_price_data("EURUSD=X", days=10)
        assert data1 == data2

    def test_generate_mock_price_data_different_per_ticker(self):
        data1 = generate_mock_price_data("EURUSD=X", days=10)
        data2 = generate_mock_price_data("GBPUSD=X", days=10)
        # They should differ (different random seeds)
        assert data1 != data2
