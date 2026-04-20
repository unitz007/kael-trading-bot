"""Tests for the REST API layer.

All endpoints are tested against the FastAPI TestClient.
Business-logic modules (ingestion, features, training) are mocked so
the tests exercise only the HTTP routing, request validation, response
formatting, and error-handling of the API layer.

Acceptance Criteria coverage
----------------------------
AC#2  – GET /api/v1/pairs  →  test_list_pairs
AC#3  – GET /api/v1/pairs/<pair>/history  →  test_get_history_success
AC#4  – POST /api/v1/pairs/<pair>/train  →  test_train_model_success
AC#5  – GET /api/v1/pairs/<pair>/predict  →  test_get_predictions_success
AC#6  – GET /api/v1/models  →  test_list_models_empty / test_list_models
AC#7  – 404 for unknown pair  →  test_history_unsupported_pair / test_train_unsupported_pair
AC#8  – 400 for invalid input  →  test_train_missing_target_column
AC#9  – 200 + JSON body for all successes  →  every 2xx test
AC#10 – delegation to existing modules  →  mocking verifies the call signatures
AC#11 – single-command entry point  →  main.py `serve` subcommand (integration)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from kael_trading_bot.api.app import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app():
    """Create a FastAPI application configured for testing."""
    return create_app()


@pytest.fixture()
def client(app):
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture()
def sample_ohlcv_df():
    """A small realistic OHLCV DataFrame for mocking ingestion."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    np.random.seed(42)
    close = 1.10 + np.cumsum(np.random.randn(100) * 0.005)
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(100) * 0.002,
            "High": close + abs(np.random.randn(100)) * 0.003,
            "Low": close - abs(np.random.randn(100)) * 0.003,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, 100),
        },
        index=dates,
    )


@pytest.fixture()
def sample_feature_df(sample_ohlcv_df):
    """OHLCV data with columns lowercased (as the API does before feature engineering)."""
    df = sample_ohlcv_df.copy()
    df.columns = [c.lower() for c in df.columns]
    return df


@pytest.fixture()
def mock_feature_matrix(sample_feature_df):
    """A feature matrix with all expected columns including target."""
    df = sample_feature_df.copy()
    # Add enough columns to mimic a real feature matrix
    for period in [10, 20, 50]:
        df[f"sma_{period}"] = df["close"].rolling(period).mean()
    for period in [9, 21]:
        df[f"ema_{period}"] = df["close"].ewm(span=period).mean()
    df["rsi_14"] = 50.0
    df["macd_12_26_9"] = 0.001
    df["macd_12_26_9_signal"] = 0.0005
    df["atr_14"] = 0.01
    df["bb_upper_20"] = df["close"] + 0.02
    df["bb_middle_20"] = df["close"]
    df["bb_lower_20"] = df["close"] - 0.02
    for w in [5, 10, 20]:
        df[f"rolling_mean_close_{w}"] = df["close"].rolling(w).mean()
        df[f"rolling_std_close_{w}"] = df["close"].rolling(w).std()
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["future_return_1"] = df["close"].pct_change(1).shift(-1)
    df["future_return_5"] = df["close"].pct_change(5).shift(-5)
    df["target_direction_1"] = (df["future_return_1"] > 0).astype(float)
    df["target_direction_5"] = (df["future_return_5"] > 0).astype(float)
    df = df.dropna()
    return df


# ---------------------------------------------------------------------------
# AC#2 — GET /api/v1/pairs
# ---------------------------------------------------------------------------


class TestListPairs:
    """Tests for the list-pairs endpoint."""

    def test_list_pairs_returns_200(self, client):
        resp = client.get("/api/v1/pairs")
        assert resp.status_code == 200

    def test_list_pairs_returns_json(self, client):
        resp = client.get("/api/v1/pairs")
        data = resp.json()
        assert data is not None
        assert isinstance(data, dict)

    def test_list_pairs_contains_pairs_key(self, client):
        resp = client.get("/api/v1/pairs")
        data = resp.json()
        assert "pairs" in data
        assert isinstance(data["pairs"], list)

    def test_list_pairs_contains_count(self, client):
        resp = client.get("/api/v1/pairs")
        data = resp.json()
        assert "count" in data
        assert data["count"] == len(data["pairs"])

    def test_list_pairs_has_known_pairs(self, client):
        resp = client.get("/api/v1/pairs")
        data = resp.json()
        assert "EURUSD=X" in data["pairs"]

    def test_list_pairs_cors_headers(self, client):
        resp = client.get("/api/v1/pairs")
        assert resp.headers.get("access-control-allow-origin") == "*"


# ---------------------------------------------------------------------------
# AC#3 — GET /api/v1/pairs/<pair>/history
# ---------------------------------------------------------------------------


class TestGetHistory:
    """Tests for the history endpoint."""

    def test_history_unsupported_pair_404(self, client):
        resp = client.get("/api/v1/pairs/INVALIDPAIR/history")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert "INVALIDPAIR" in data["error"]

    @patch("kael_trading_bot.api.app.ForexDataFetcher")
    def test_history_success(self, mock_fetcher_cls, client, sample_ohlcv_df):
        mock_fetcher = MagicMock()
        mock_fetcher.get.return_value = sample_ohlcv_df
        mock_fetcher_cls.return_value = mock_fetcher

        resp = client.get("/api/v1/pairs/EURUSD=X/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pair"] == "EURUSD=X"
        assert data["rows"] == len(sample_ohlcv_df)
        assert isinstance(data["data"], list)
        assert len(data["data"]) > 0

    @patch("kael_trading_bot.api.app.ForexDataFetcher")
    def test_history_data_has_ohlcv_fields(self, mock_fetcher_cls, client, sample_ohlcv_df):
        mock_fetcher = MagicMock()
        mock_fetcher.get.return_value = sample_ohlcv_df
        mock_fetcher_cls.return_value = mock_fetcher

        resp = client.get("/api/v1/pairs/EURUSD=X/history")
        data = resp.json()
        first_row = data["data"][0]
        for field in ("Date", "Open", "High", "Low", "Close", "Volume"):
            assert field in first_row, f"Missing {field} in response"

    @patch("kael_trading_bot.api.app.ForexDataFetcher")
    def test_history_pair_without_suffix_normalised(self, mock_fetcher_cls, client, sample_ohlcv_df):
        """Passing 'EURUSD' (without =X) should still work."""
        mock_fetcher = MagicMock()
        mock_fetcher.get.return_value = sample_ohlcv_df
        mock_fetcher_cls.return_value = mock_fetcher

        resp = client.get("/api/v1/pairs/EURUSD/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pair"] == "EURUSD=X"

    @patch("kael_trading_bot.api.app.ForexDataFetcher")
    def test_history_empty_data_404(self, mock_fetcher_cls, client):
        mock_fetcher = MagicMock()
        mock_fetcher.get.side_effect = ValueError("No data returned for pair")
        mock_fetcher_cls.return_value = mock_fetcher

        resp = client.get("/api/v1/pairs/EURUSD=X/history")
        assert resp.status_code == 404

    @patch("kael_trading_bot.api.app.ForexDataFetcher")
    def test_history_internal_error_500(self, mock_fetcher_cls, client):
        mock_fetcher = MagicMock()
        mock_fetcher.get.side_effect = RuntimeError("network failure")
        mock_fetcher_cls.return_value = mock_fetcher

        resp = client.get("/api/v1/pairs/EURUSD=X/history")
        assert resp.status_code == 500

    def test_history_options_preflight(self, client):
        resp = client.options("/api/v1/pairs/EURUSD=X/history")
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "*"


# ---------------------------------------------------------------------------
# AC#4 — POST /api/v1/pairs/<pair>/train
# ---------------------------------------------------------------------------


class TestTrainModel:
    """Tests for the training endpoint."""

    def test_train_unsupported_pair_404(self, client):
        resp = client.post("/api/v1/pairs/INVALIDPAIR/train")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data

    @patch("kael_trading_bot.api.app.TrainingPipeline")
    @patch("kael_trading_bot.api.app.build_feature_matrix")
    @patch("kael_trading_bot.api.app.ForexDataFetcher")
    def test_train_success(
        self, mock_fetcher_cls, mock_build_features, mock_pipeline_cls,
        client, sample_ohlcv_df, mock_feature_matrix,
    ):
        # Setup mocks
        mock_fetcher = MagicMock()
        mock_fetcher.get.return_value = sample_ohlcv_df
        mock_fetcher_cls.return_value = mock_fetcher

        mock_build_features.return_value = mock_feature_matrix

        mock_result = MagicMock()
        mock_result.model_name = "eurusd_x"
        mock_result.model_version = "v20240101T000000"
        mock_result.model_type = "xgboost"
        mock_result.test_eval = MagicMock()
        mock_result.test_eval.to_dict.return_value = {"accuracy": 0.55}
        mock_result.saved_path = "/models/eurusd_x/v20240101T000000"

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_result
        mock_pipeline_cls.return_value = mock_pipeline

        resp = client.post("/api/v1/pairs/EURUSD=X/train")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pair"] == "EURUSD=X"
        assert data["status"] == "completed"
        assert data["model_name"] == "eurusd_x"
        assert "duration_seconds" in data
        assert "samples_trained" in data
        assert "num_features" in data
        assert data["test_metrics"] == {"accuracy": 0.55}

    @patch("kael_trading_bot.api.app.TrainingPipeline")
    @patch("kael_trading_bot.api.app.build_feature_matrix")
    @patch("kael_trading_bot.api.app.ForexDataFetcher")
    def test_train_missing_target_column_400(
        self, mock_fetcher_cls, mock_build_features, mock_pipeline_cls,
        client, sample_ohlcv_df, sample_feature_df,
    ):
        """When build_feature_matrix returns a df without the target column → 400."""
        mock_fetcher = MagicMock()
        mock_fetcher.get.return_value = sample_ohlcv_df
        mock_fetcher_cls.return_value = mock_fetcher

        # Return df without target column
        mock_build_features.return_value = sample_feature_df

        resp = client.post("/api/v1/pairs/EURUSD=X/train")
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data
        assert "target_direction_1" in data["error"]

    @patch("kael_trading_bot.api.app.TrainingPipeline")
    @patch("kael_trading_bot.api.app.build_feature_matrix")
    @patch("kael_trading_bot.api.app.ForexDataFetcher")
    def test_train_pair_without_suffix(
        self, mock_fetcher_cls, mock_build_features, mock_pipeline_cls,
        client, sample_ohlcv_df, mock_feature_matrix,
    ):
        mock_fetcher = MagicMock()
        mock_fetcher.get.return_value = sample_ohlcv_df
        mock_fetcher_cls.return_value = mock_fetcher

        mock_build_features.return_value = mock_feature_matrix

        mock_result = MagicMock()
        mock_result.model_name = "eurusd_x"
        mock_result.model_version = "v20240101T000000"
        mock_result.model_type = "xgboost"
        mock_result.test_eval = None
        mock_result.saved_path = None

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_result
        mock_pipeline_cls.return_value = mock_pipeline

        resp = client.post("/api/v1/pairs/EURUSD/train")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pair"] == "EURUSD=X"

    @patch("kael_trading_bot.api.app.TrainingPipeline")
    @patch("kael_trading_bot.api.app.build_feature_matrix")
    @patch("kael_trading_bot.api.app.ForexDataFetcher")
    def test_train_delegates_to_pipeline(
        self, mock_fetcher_cls, mock_build_features, mock_pipeline_cls,
        client, sample_ohlcv_df, mock_feature_matrix,
    ):
        """Verify the training endpoint delegates to TrainingPipeline (AC#10)."""
        mock_fetcher = MagicMock()
        mock_fetcher.get.return_value = sample_ohlcv_df
        mock_fetcher_cls.return_value = mock_fetcher

        mock_build_features.return_value = mock_feature_matrix

        mock_result = MagicMock()
        mock_result.model_name = "eurusd_x"
        mock_result.model_version = "v1"
        mock_result.model_type = "xgboost"
        mock_result.test_eval = None
        mock_result.saved_path = None

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_result
        mock_pipeline_cls.return_value = mock_pipeline

        resp = client.post("/api/v1/pairs/EURUSD=X/train")
        assert resp.status_code == 200
        mock_pipeline.run.assert_called_once()

        # Verify X and y were passed correctly
        call_args = mock_pipeline.run.call_args
        X = call_args[0][0]
        y = call_args[0][1]
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    @patch("kael_trading_bot.api.app.build_feature_matrix")
    @patch("kael_trading_bot.api.app.ForexDataFetcher")
    def test_train_ingestion_error_400(
        self, mock_fetcher_cls, mock_build_features, client,
    ):
        mock_fetcher = MagicMock()
        mock_fetcher.get.side_effect = ValueError("No data")
        mock_fetcher_cls.return_value = mock_fetcher

        resp = client.post("/api/v1/pairs/EURUSD=X/train")
        assert resp.status_code == 400

    @patch("kael_trading_bot.api.app.TrainingPipeline")
    @patch("kael_trading_bot.api.app.build_feature_matrix")
    @patch("kael_trading_bot.api.app.ForexDataFetcher")
    def test_train_internal_error_500(
        self, mock_fetcher_cls, mock_build_features, mock_pipeline_cls,
        client, sample_ohlcv_df, mock_feature_matrix,
    ):
        mock_fetcher = MagicMock()
        mock_fetcher.get.return_value = sample_ohlcv_df
        mock_fetcher_cls.return_value = mock_fetcher

        mock_build_features.return_value = mock_feature_matrix

        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = RuntimeError("OOM")
        mock_pipeline_cls.return_value = mock_pipeline

        resp = client.post("/api/v1/pairs/EURUSD=X/train")
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# AC#5 — GET /api/v1/pairs/<pair>/predict
# ---------------------------------------------------------------------------


class TestGetPredictions:
    """Tests for the prediction endpoint."""

    def test_predict_no_model_404(self, client):
        """When no trained model exists for the pair → 404."""
        with patch("kael_trading_bot.api.app.ModelPersistence") as mock_cls:
            mock_persistence = MagicMock()
            mock_persistence.list_versions.return_value = []
            mock_cls.return_value = mock_persistence

            resp = client.get("/api/v1/pairs/EURUSD=X/predict")
            assert resp.status_code == 404
            data = resp.json()
            assert "error" in data
            assert "Train a model first" in data["error"]

    @patch("kael_trading_bot.api.app.build_feature_matrix")
    @patch("kael_trading_bot.api.app.ForexDataFetcher")
    @patch("kael_trading_bot.api.app.ModelPersistence")
    def test_predict_success(
        self, mock_persistence_cls, mock_fetcher_cls, mock_build_features,
        client, sample_ohlcv_df, mock_feature_matrix,
    ):
        # Persistence mock — returns a model + metadata
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
        mock_model.predict_proba.return_value = np.array(
            [[0.6, 0.4], [0.3, 0.7], [0.4, 0.6], [0.7, 0.3], [0.2, 0.8]]
        )

        mock_metadata = MagicMock()
        mock_metadata.trained_at = "2024-01-01T00:00:00+00:00"
        mock_metadata.feature_names = [
            c for c in mock_feature_matrix.columns
            if not c.startswith(("future_return", "target_"))
        ]

        mock_persistence = MagicMock()
        mock_persistence.list_versions.return_value = ["v20240101T000000"]
        mock_persistence.load.return_value = (mock_model, mock_metadata)
        mock_persistence_cls.return_value = mock_persistence

        # Ingestion mock
        mock_fetcher = MagicMock()
        mock_fetcher.get.return_value = sample_ohlcv_df
        mock_fetcher_cls.return_value = mock_fetcher

        # Feature mock — return 5 rows to match prediction array
        small_df = mock_feature_matrix.iloc[:5]
        mock_build_features.return_value = small_df

        resp = client.get("/api/v1/pairs/EURUSD=X/predict")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pair"] == "EURUSD=X"
        assert data["model_name"] == "eurusd_x"
        assert data["model_version"] == "v20240101T000000"
        assert data["total_predictions"] == 5
        assert data["up_count"] == 3
        assert data["down_count"] == 2
        assert len(data["predictions"]) == 5
        assert "date" in data["predictions"][0]
        assert "prediction" in data["predictions"][0]
        assert "probability_up" in data["predictions"][0]

    @patch("kael_trading_bot.api.app.ModelPersistence")
    def test_predict_no_feature_names_400(self, mock_persistence_cls, client):
        """When model metadata has no feature names → 400."""
        mock_model = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.feature_names = None

        mock_persistence = MagicMock()
        mock_persistence.list_versions.return_value = ["v1"]
        mock_persistence.load.return_value = (mock_model, mock_metadata)
        mock_persistence_cls.return_value = mock_persistence

        resp = client.get("/api/v1/pairs/EURUSD=X/predict")
        assert resp.status_code == 400
        data = resp.json()
        assert "no feature names" in data["error"]

    @patch("kael_trading_bot.api.app.build_feature_matrix")
    @patch("kael_trading_bot.api.app.ForexDataFetcher")
    @patch("kael_trading_bot.api.app.ModelPersistence")
    def test_predict_missing_features_400(
        self, mock_persistence_cls, mock_fetcher_cls, mock_build_features,
        client, sample_ohlcv_df,
    ):
        """When the feature matrix doesn't have the expected features → 400."""
        mock_model = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.trained_at = "2024-01-01T00:00:00+00:00"
        mock_metadata.feature_names = ["nonexistent_feature"]

        mock_persistence = MagicMock()
        mock_persistence.list_versions.return_value = ["v1"]
        mock_persistence.load.return_value = (mock_model, mock_metadata)
        mock_persistence_cls.return_value = mock_persistence

        mock_fetcher = MagicMock()
        mock_fetcher.get.return_value = sample_ohlcv_df
        mock_fetcher_cls.return_value = mock_fetcher

        df = sample_ohlcv_df.copy()
        df.columns = [c.lower() for c in df.columns]
        mock_build_features.return_value = df

        resp = client.get("/api/v1/pairs/EURUSD=X/predict")
        assert resp.status_code == 400
        data = resp.json()
        assert "Missing features" in data["error"]

    @patch("kael_trading_bot.api.app.ModelPersistence")
    def test_predict_model_load_error_500(self, mock_persistence_cls, client):
        mock_persistence = MagicMock()
        mock_persistence.list_versions.side_effect = RuntimeError("db error")
        mock_persistence_cls.return_value = mock_persistence

        resp = client.get("/api/v1/pairs/EURUSD=X/predict")
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# AC#6 — GET /api/v1/models
# ---------------------------------------------------------------------------


class TestListModels:
    """Tests for the list-models endpoint."""

    def test_list_models_empty(self, client):
        with patch("kael_trading_bot.api.app.ModelPersistence") as mock_cls:
            mock_persistence = MagicMock()
            mock_persistence.list_models.return_value = []
            mock_cls.return_value = mock_persistence

            resp = client.get("/api/v1/models")
            assert resp.status_code == 200
            data = resp.json()
            assert data["models"] == []
            assert data["count"] == 0

    def test_list_models_with_data(self, client):
        with patch("kael_trading_bot.api.app.ModelPersistence") as mock_cls:
            mock_meta = MagicMock()
            mock_meta.model_type = "xgboost"
            mock_meta.trained_at = "2024-01-15T10:30:00+00:00"
            mock_meta.metrics = {"accuracy": 0.55}
            mock_meta.feature_names = ["sma_10", "rsi_14"]

            mock_persistence = MagicMock()
            mock_persistence.list_models.return_value = ["eurusd_x"]
            mock_persistence.list_versions.return_value = ["v20240115T103000"]
            mock_persistence.load.return_value = (MagicMock(), mock_meta)
            mock_cls.return_value = mock_persistence

            resp = client.get("/api/v1/models")
            assert resp.status_code == 200
            data = resp.json()
            assert data["count"] == 1
            model_info = data["models"][0]
            assert model_info["model_name"] == "eurusd_x"
            assert model_info["version"] == "v20240115T103000"
            assert model_info["model_type"] == "xgboost"
            assert model_info["trained_at"] == "2024-01-15T10:30:00+00:00"
            assert model_info["metrics"] == {"accuracy": 0.55}

    def test_list_models_skips_corrupted(self, client):
        """Corrupted model versions should be skipped gracefully."""
        with patch("kael_trading_bot.api.app.ModelPersistence") as mock_cls:
            mock_persistence = MagicMock()
            mock_persistence.list_models.return_value = ["eurusd_x"]
            mock_persistence.list_versions.return_value = ["v1", "v2_corrupt"]
            mock_persistence.load.side_effect = [
                (MagicMock(), MagicMock()),
                FileNotFoundError("missing"),
            ]
            mock_cls.return_value = mock_persistence

            resp = client.get("/api/v1/models")
            assert resp.status_code == 200
            data = resp.json()
            assert data["count"] == 1

    def test_list_models_internal_error_500(self, client):
        with patch("kael_trading_bot.api.app.ModelPersistence") as mock_cls:
            mock_persistence = MagicMock()
            mock_persistence.list_models.side_effect = RuntimeError("db error")
            mock_cls.return_value = mock_persistence

            resp = client.get("/api/v1/models")
            assert resp.status_code == 500


# ---------------------------------------------------------------------------
# AC#11 — CLI entry point
# ---------------------------------------------------------------------------


class TestServeCommand:
    """Tests for the `serve` CLI subcommand."""

    @patch("main.uvicorn.run")
    def test_serve_starts_uvicorn(self, mock_run):
        """Verify that cmd_serve calls uvicorn.run with the configured port."""
        from main import cmd_serve

        cmd_serve(port=8080)
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs[0][0] is not None  # app object
        assert call_kwargs[1]["host"] == "0.0.0.0"
        assert call_kwargs[1]["port"] == 8080

    @patch("main.uvicorn.run")
    def test_serve_default_port(self, mock_run):
        from main import cmd_serve

        cmd_serve(port=5000)
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs[1]["port"] == 5000


# ---------------------------------------------------------------------------
# Helper / normalisation tests
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for internal helper functions."""

    def test_normalise_ticker_adds_suffix(self):
        from kael_trading_bot.api.app import _normalise_ticker

        assert _normalise_ticker("EURUSD") == "EURUSD=X"
        assert _normalise_ticker("EURUSD=X") == "EURUSD=X"

    def test_pair_to_model_name(self):
        from kael_trading_bot.api.app import _pair_to_model_name

        assert _pair_to_model_name("EURUSD=X") == "eurusd_x"
        assert _pair_to_model_name("^GSPC") == "_gspc"

    def test_is_supported_pair(self):
        from kael_trading_bot.api.app import _is_supported_pair

        assert _is_supported_pair("EURUSD=X") is True
        assert _is_supported_pair("EURUSD") is True
        assert _is_supported_pair("INVALID=X") is False
