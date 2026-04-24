# Kael Trading Bot — Technical Overview

## Repository Information

- **Name**: [unitz007/kael-trading-bot](https://github.com/unitz007/kael-trading-bot)

## Tech Stack

- **Backend Logic**: Python (3.11+)
- **Frontend**: JavaScript, HTML, CSS (React-based)
- **API Framework**: FastAPI (Python)
- **Machine Learning Libraries**: scikit-learn, XGBoost, LightGBM
- **Data Handling**: pandas, numpy, yfinance
- **Containerization**: Docker, Docker Compose

## Core Directories and Files

- `main.py` — CLI entry point for training, predicting, and serving.
- `src/` — Contains the core Python modules:
  - `kael_trading_bot/`: Main package with ingestion, features, training, and API modules.
- `frontend/` — Houses the React-based web UI.
- `tests/` — Unit and integration tests for backend logic.
- `Dockerfile` — Defines the container image for the backend service.
- `docker-compose.yml` — Orchestrates multi-container deployments (backend + frontend).
- `requirements.txt` — Lists Python dependencies.
- `pyproject.toml` — Package metadata and configuration.

## Containerization Strategy

The project is fully containerized using Docker:

- **Dockerfile**: Builds a lean Python image with a multi-stage approach for efficient dependency management.
- **docker-compose.yml**: Coordinates backend (FastAPI) and frontend (Nginx-hosted React) services.
  - Exposes backend on port 5000
  - Exposes frontend on port 3000
  - Volumes are mounted for persistent storage (`models/`, `.cache/forex_data`, `logs/`)
- **Environment Variables**: Configurable via `.env` or `docker-compose` overrides.

## Primary Features

- **Broker API Integration**: Pulls forex data from Yahoo Finance using `yfinance`.
- **Market Data Parsing**: Processes OHLCV data to extract key financial indicators.
- **Technical Feature Engineering**: Computes a wide variety of technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, rolling stats, temporal features).
- **Machine Learning Model Training**: Supports classification models (XGBoost, LightGBM, Random Forest, Logistic Regression) to predict market direction.
- **Trading Signals & Execution**: Generates actionable trade setups (entry, stop-loss, take-profit) based on model outputs.
- **Activity Logging**: Tracks training runs and stores model metadata for auditability.
- **REST API**: Exposes bot capabilities over HTTP (data retrieval, training triggers, predictions, forecasts, trade setups).
- **Web UI**: Allows users to browse pairs, train models, and view predictions in a browser-based dashboard.