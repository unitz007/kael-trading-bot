# Kael Trading Bot

An ML-based forex trading bot that uses machine learning models to analyze market data, identify trading opportunities, and execute trades on forex markets.

## Overview

Kael Trading Bot is a Python project that builds a machine learning pipeline for forex trading. It fetches historical OHLCV (Open, High, Low, Close, Volume) data, engineers technical and temporal features, trains classification models to predict price direction, and evaluates them with both statistical and trading-oriented metrics.

## Modules

### 1. Forex Data Ingestion

Fetches, validates, and caches historical OHLCV forex data from Yahoo Finance via `yfinance`.

- Downloads data for configurable currency pairs, date ranges, and intervals
- Validates column types, chronological ordering, and OHLCV logic (e.g. High ≥ Low)
- Handles missing values with configurable strategies (forward-fill, drop, or raise)
- Caches fetched data as Parquet files for fast subsequent loads

**Key file:** `src/kael_trading_bot/ingestion.py`

### 2. Technical Feature Engineering

Transforms raw OHLCV data into a rich feature matrix for ML model consumption.

- **Technical indicators:** SMA, EMA, RSI, MACD (line, signal, histogram), Bollinger Bands (upper, middle, lower, bandwidth, %B), and ATR
- **Temporal features:** day of week, hour of day, minute of hour, month start/end flags
- **Rolling statistics:** rolling mean, std, min, max, and skew over configurable windows
- **Target variables:** future log/simple returns and categorical direction labels (up/down/flat)
- Configurable via a single `FeatureConfig` dataclass

**Key files:** `src/kael_trading_bot/features/` (`indicators.py`, `temporal.py`, `targets.py`, `pipeline.py`)

### 3. ML Model Training Pipeline

End-to-end training pipeline with time-aware splitting, multiple model types, and full observability.

- **Supported models:** XGBoost, LightGBM, Random Forest, Logistic Regression — all with sensible defaults and overridable hyper-parameters
- **Time-aware splitting:** chronological train/val/test split and expanding-window cross-validation (no future data leakage)
- **Evaluation:** classification metrics (accuracy, precision, recall, F1, ROC-AUC) and trading-oriented metrics (hit rate, average return per trade, Sharpe ratio, max drawdown)
- **Model persistence:** save/load trained models with JSON metadata sidecars (model type, version, params, metrics)
- **Training logging:** every run is recorded to a JSON-lines log file for reproducibility and auditing

**Key files:** `src/kael_trading_bot/training/` (`pipeline.py`, `models.py`, `splitting.py`, `evaluation.py`, `persistence.py`, `logging.py`)

### 4. CI Pipeline

GitHub Actions workflow that runs the test suite on every pull request to catch regressions early.

**Key file:** `.github/workflows/ci.yml`

### 5. Project Scaffolding

Standard Python package structure with dependency management and tooling configuration.

- Package layout under `src/kael_trading_bot/`
- Dependencies declared in `pyproject.toml` with optional dev dependencies (pytest, ruff, mypy)
- Editable install support via `pip install -e .`

## Architecture

```
src/kael_trading_bot/
├── __init__.py            # Package entry point
├── config.py              # Central configuration (pairs, dates, cache settings)
├── ingestion.py           # Data fetching, validation, and caching
├── features/
│   ├── __init__.py        # Package init, re-exports build_feature_matrix
│   ├── indicators.py      # Technical indicator computations
│   ├── temporal.py        # Time-based and rolling-window features
│   ├── targets.py         # Target variable generation (returns, direction)
│   └── pipeline.py        # Orchestrates full feature engineering pipeline
└── training/
    ├── __init__.py        # Package init
    ├── models.py           # Model registry and factory
    ├── splitting.py        # Time-aware train/val/test splitting
    ├── evaluation.py       # Classification and trading metrics
    ├── persistence.py      # Model save/load with metadata
    ├── logging.py          # Training run logger (JSON-lines)
    └── pipeline.py         # End-to-end training pipeline
```

**Data flow:**

1. **Ingestion** fetches raw OHLCV data for configured forex pairs
2. **Feature Engineering** transforms raw data into a feature matrix with indicators, temporal features, and target variables
3. **Training Pipeline** splits data chronologically, trains a model, evaluates it, and persists the result

## Prerequisites

- **Python 3.11+** (developed and tested with 3.11)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/unitz007/kael-trading-bot.git
cd kael-trading-bot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -e ".[dev]"
```

### 4. Verify the installation

```bash
python -c "import kael_trading_bot; print('Package installed successfully!')"
```

## Development

The package is installed in editable mode for development:

```bash
pip install -e .
```

Run tests with:

```bash
pytest
```

## License

TBD
