"""Tests for the web UI REST API routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from kael_trading_bot.web.app import create_app
from kael_trading_bot.web.jobs import get_job_store


@pytest.fixture()
def client():
    """Create a test client and reset job store between tests."""
    app = create_app()
    get_job_store().clear()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------


class TestPageRoutes:
    def test_root_redirects_to_training(self, client: TestClient):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 307
        assert resp.headers["location"] == "/training"

    def test_training_page_renders(self, client: TestClient):
        resp = client.get("/training")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Model Training" in resp.text


# ---------------------------------------------------------------------------
# API — pairs
# ---------------------------------------------------------------------------


class TestPairsAPI:
    def test_list_pairs(self, client: TestClient):
        resp = client.get("/api/pairs")
        assert resp.status_code == 200
        data = resp.json()
        assert "pairs" in data
        assert isinstance(data["pairs"], list)
        assert len(data["pairs"]) > 0
        # Check structure
        pair = data["pairs"][0]
        assert "ticker" in pair
        assert "display" in pair

    def test_pairs_include_eurusd(self, client: TestClient):
        resp = client.get("/api/pairs")
        tickers = [p["ticker"] for p in resp.json()["pairs"]]
        assert "EURUSD=X" in tickers


# ---------------------------------------------------------------------------
# API — training
# ---------------------------------------------------------------------------


class TestTrainingAPI:
    def test_list_training_status_empty(self, client: TestClient):
        resp = client.get("/api/training/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "jobs" in data
        assert data["jobs"] == []

    def test_get_nonexistent_job(self, client: TestClient):
        resp = client.get("/api/training/status/nonexistent-id")
        assert resp.status_code == 404

    def test_start_training_returns_job(self, client: TestClient):
        resp = client.post("/api/training/start?pair=EURUSD")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pair"] == "EURUSD=X"
        assert data["status"] == "running"
        assert "job_id" in data


# ---------------------------------------------------------------------------
# API — models
# ---------------------------------------------------------------------------


class TestModelsAPI:
    def test_list_models(self, client: TestClient):
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)
