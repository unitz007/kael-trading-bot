"""Tests for the web predictions page and API.

Covers:
- PredictionService unit tests (model status, prediction generation)
- FastAPI route tests (page rendering, API endpoints)
- Edge cases (no model found, missing features)
"""

from __future__ import annotations


from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.kael_trading_bot.training.persistence import ModelMetadata, ModelPersistence
from src.kael_trading_bot.web.app import app
from src.kael_trading_bot.web.predictions import (
    ModelStatus,
    PredictionResult,
    PredictionService,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_models_dir(tmp_path: Path) -> Path:
    """Create a temporary models directory with a sample trained model."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir


@pytest.fixture
def sample_metadata() -> ModelMetadata:
    """Return sample model metadata."""
    return ModelMetadata(
        model_type="xgboost",
        model_version="v20240101T120000",
        params={"n_estimators": 200},
        metrics={},
        feature_names=["sma_10", "rsi_14", "macd"],
        trained_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc).isoformat(),
    )


@pytest.fixture
def persistence_with_model(
    tmp_models_dir: Path,
    sample_metadata: ModelMetadata,
) -> tuple[ModelPersistence, str]:
    """Create a ModelPersistence with one saved model version."""
    import joblib
    from sklearn.dummy import DummyClassifier

    persistence = ModelPersistence(directory=str(tmp_models_dir))

    # Save a dummy model
    model = DummyClassifier(strategy="constant", constant=1)
    model.fit(np.array([[1, 2, 3]]), np.array([1]))

    model_dir = persistence.save(
        model, "eurusd_x", "v20240101T120000", sample_metadata
    )
    return persistence, "eurusd_x"


@pytest.fixture
def service(persistence_with_model: tuple[ModelPersistence, str]) -> PredictionService:
    """Create a PredictionService backed by a mock persistence."""
    persistence, _ = persistence_with_model
    return PredictionService(persistence=persistence)


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# PredictionService Tests
# ---------------------------------------------------------------------------


class TestPredictionService:
    """Unit tests for PredictionService."""

    def test_list_available_pairs(self, service: PredictionService) -> None:
        """Should return the configured forex pairs."""
        pairs = service.list_available_pairs()
        assert isinstance(pairs, list)
        assert len(pairs) > 0
        assert "EURUSD=X" in pairs

    def test_get_model_status_available(
        self,
        persistence_with_model: tuple[ModelPersistence, str],
    ) -> None:
        """Should report model as available when it exists."""
        persistence, model_name = persistence_with_model
        svc = PredictionService(persistence=persistence)

        # The model_name is eurusd_x, corresponding to EURUSD=X
        status = svc.get_model_status("EURUSD=X")

        assert isinstance(status, ModelStatus)
        assert status.available is True
        assert status.latest_version == "v20240101T120000"
        assert len(status.versions) == 1

    def test_get_model_status_unavailable(self, service: PredictionService) -> None:
        """Should report model as unavailable when it doesn't exist."""
        # GBPUSD should not have a model in our fixture
        status = service.get_model_status("GBPUSD=X")
        assert status.available is False
        assert status.versions == []
        assert status.latest_version is None

    def test_get_all_model_statuses(self, service: PredictionService) -> None:
        """Should return statuses for all configured pairs."""
        statuses = service.get_all_model_statuses()
        assert len(statuss) > 1
        # At least one should be available (eurusd_x)
        available = [s for s in statuses if s.available]
        assert len(available) >= 1

    def test_predict_no_model(self, service: PredictionService) -> None:
        """Should raise FileNotFoundError when no model exists."""
        with pytest.raises(FileNotFoundError, match="No trained model found"):
            service.predict("GBPUSD=X")

    def test_predict_with_model(
        self,
        persistence_with_model: tuple[ModelPersistence, str],
        sample_metadata: ModelMetadata,
    ) -> None:
        """Should generate a prediction when model exists."""
        persistence, _ = persistence_with_model
        svc = PredictionService(persistence=persistence)

        # Mock the ingestion + feature engineering parts
        mock_raw_df = MagicMock()
        mock_raw_df.columns = ["Open", "High", "Low", "Close"]

        mock_feature_df = MagicMock()
        mock_feature_df.columns = sample_metadata.feature_names
        mock_feature_df.values = np.array([[1.0, 2.0, 3.0]])
        mock_feature_df.index = [0]

        with patch(
            "src.kael_trading_bot.web.predictions.ForexDataFetcher"
        ) as MockFetcher, patch(
            "src.kael_trading_bot.web.predictions.build_feature_matrix"
        ) as MockBuild:
            MockFetcher.return_value.get.return_value = mock_raw_df
            MockBuild.return_value = mock_feature_df

            result = svc.predict("EURUSD=X")

        assert isinstance(result, PredictionResult)
        assert result.pair == "EURUSD=X"
        assert result.direction in ("UP", "DOWN")
        assert 0.0 <= result.confidence <= 1.0
        assert result.model_version == "v20240101T120000"
        assert result.model_type == "xgboost"
        assert result.generated_at is not None


# ---------------------------------------------------------------------------
# FastAPI Route Tests
# ---------------------------------------------------------------------------


class TestPredictionsPage:
    """Integration tests for the predictions web page."""

    def test_index_redirects_to_predictions(self, client: TestClient) -> None:
        """Root URL should redirect to /predictions."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert "/predictions" in response.headers["location"]

    def test_predictions_page_renders(self, client: TestClient) -> None:
        """The predictions page should render with status 200."""
        response = client.get("/predictions")
        assert response.status_code == 200
        html = response.text
        # Check that key elements exist
        assert "Predictions" in html
        assert "EURUSD" in html
        assert "pair-grid" in html

    def test_predictions_page_shows_all_pairs(self, client: TestClient) -> None:
        """The page should list all configured forex pairs."""
        response = client.get("/predictions")
        html = response.text
        expected_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]
        for pair in expected_pairs:
            assert pair in html


class TestPredictionsAPI:
    """Integration tests for the predictions API endpoints."""

    def test_api_status_returns_json(self, client: TestClient) -> None:
        """GET /api/predictions/status should return JSON."""
        response = client.get("/api/predictions/status")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        # Each item should have expected fields
        for item in data:
            assert "pair" in item
            assert "display_name" in item
            assert "model_available" in item

    def test_api_predict_missing_pair(self, client: TestClient) -> None:
        """POST /api/predictions without pair should return 400."""
        response = client.post(
            "/api/predictions",
            json={},
        )
        assert response.status_code == 400
        assert "Missing" in response.json()["detail"]

    def test_api_predict_no_model(self, client: TestClient) -> None:
        """POST /api/predictions for a pair without a model should return 404."""
        response = client.post(
            "/api/predictions",
            json={"pair": "GBPUSD=X"},
        )
        assert response.status_code == 404
        assert "No trained model" in response.json()["detail"]

    def test_api_predict_with_model(
        self,
        persistence_with_model: tuple[ModelPersistence, str],
        sample_metadata: ModelMetadata,
    ) -> None:
        """POST /api/predictions should return prediction result when model exists."""
        from src.kael_trading_bot.web.app import service

        # Replace the service with one backed by our test persistence
        persistence, _ = persistence_with_model
        service.persistence = persistence

        client = TestClient(app)

        # Mock the heavy data-fetching parts
        mock_raw_df = MagicMock()
        mock_raw_df.columns = ["Open", "High", "Low", "Close"]

        mock_feature_df = MagicMock()
        mock_feature_df.columns = sample_metadata.feature_names
        mock_feature_df.values = np.array([[1.0, 2.0, 3.0]])
        mock_feature_df.index = [0]

        with patch(
            "src.kael_trading_bot.web.predictions.ForexDataFetcher"
        ) as MockFetcher, patch(
            "src.kael_trading_bot.web.predictions.build_feature_matrix"
        ) as MockBuild:
            MockFetcher.return_value.get.return_value = mock_raw_df
            MockBuild.return_value = mock_feature_df

            response = client.post(
                "/api/predictions",
                json={"pair": "EURUSD=X"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["pair"] == "EURUSD=X"
        assert data["display_name"] == "EURUSD"
        assert data["direction"] in ("UP", "DOWN")
        assert isinstance(data["confidence"], float)
        assert data["model_version"] == "v20240101T120000"
        assert "generated_at" in data

    def test_api_predict_invalid_json(self, client: TestClient) -> None:
        """POST /api/predictions with invalid JSON should return 400."""
        response = client.post(
            "/api/predictions",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400