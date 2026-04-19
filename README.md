# Kael Trading Bot

ML-based forex trading bot with technical feature engineering and model training pipeline.

---

## Table of Contents

- [Overview](#overview)
- [Usage](#usage)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Model Training](#model-training)
  - [Using a Trained Model](#using-a-trained-model)
  - [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

Kael Trading Bot ingests forex pair data via Yahoo Finance, engineers technical features (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, rolling stats, temporal features, and directional targets), trains ML classification models (XGBoost, LightGBM, Random Forest, Logistic Regression), evaluates them with both classification and trading-oriented metrics, and persists trained models for downstream use.

---

## Usage

### Prerequisites

| Requirement          | Details                                                        |
| -------------------- | -------------------------------------------------------------- |
| **Python**           | 3.11 or higher (see `pyproject.toml`)                          |
| **Operating system** | Linux, macOS, or Windows (WSL recommended on Windows)          |
| **Data source**      | Internet access to [Yahoo Finance](https://finance.yahoo.com) via the `yfinance` library — no API key required |
| **External tools**   | Git (for cloning the repository)                               |

> **Note:** Yahoo Finance provides free forex data without authentication. No API key is needed for the default data ingestion pipeline.

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/unitz007/kael-trading-bot.git
   cd kael-trading-bot
   ```

2. **(Recommended) Create and activate a virtual environment:**

   ```bash
   # Using venv
   python -m venv .venv
   source .venv/bin/activate   # Linux / macOS
   .venv\Scripts\activate       # Windows

   # Or using conda
   conda create -n kael-bot python=3.11
   conda activate kael-bot
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   Key dependencies include:

   - `numpy`, `pandas` — data handling
   - `scikit-learn`, `xgboost`, `lightgbm` — ML model training
   - `yfinance` — forex data ingestion from Yahoo Finance
   - `ta` — technical analysis indicators
   - `pyarrow` — Parquet I/O for data caching
   - `joblib` — model persistence
   - `matplotlib`, `seaborn` — visualisation
   - `python-dotenv` — environment variable loading

4. **(Optional) Install the package in editable mode for development:**

   ```bash
   pip install -e ".[dev]"
   ```

   This also installs dev dependencies (`pytest`, `pytest-cov`, `ruff`, `mypy`).

### Configuration

The project is configured through **Python dataclasses** and **environment variables**. There is no YAML config file — all settings are set programmatically.

#### Environment Variables

| Variable            | Description                              | Default                |
| ------------------- | ---------------------------------------- | ---------------------- |
| `KAEL_START_DATE`   | Data start date (ISO 8601, inclusive)    | `2020-01-01`           |
| `KAEL_END_DATE`     | Data end date (ISO 8601, inclusive)      | `2025-01-01`           |
| `KAEL_INTERVAL`     | Data frequency (`1d`, `1h`, etc.)        | `1d`                   |
| `KAEL_CACHE_DIR`    | Directory for cached Parquet data files   | `.cache/forex_data`    |

These are read by `IngestionConfig` (in `src/kael_trading_bot/config.py`). Set them in your shell or a `.env` file loaded with `python-dotenv`.

#### Key Configuration Classes

| Class             | Module / File                                              | Purpose                                        |
| ----------------- | ---------------------------------------------------------- | ---------------------------------------------- |
| `IngestionConfig` | `kael_trading_bot.config` (or `src.kael_trading_bot.config`) | Forex pairs, date ranges, interval, cache dir  |
| `FeatureConfig`   | `kael_trading_bot.features.pipeline`                        | Indicator windows, target horizons, NaN policy |
| `PipelineConfig`  | `src.kael_trading_bot.training.pipeline`                    | Model type, split ratios, CV, persistence      |

Example — override defaults programmatically:

```python
from kael_trading_bot.config import IngestionConfig
from kael_trading_bot.features.pipeline import FeatureConfig
from src.kael_trading_bot.training.pipeline import PipelineConfig

ingestion_cfg = IngestionConfig(
    pairs=("EURUSD=X", "GBPUSD=X"),
    start_date="2022-01-01",
    end_date="2024-12-31",
)

feature_cfg = FeatureConfig(
    sma_periods=(10, 20, 50),
    rsi_period=14,
    target_horizons=[1, 5],
)

pipeline_cfg = PipelineConfig(
    model_type="xgboost",
    model_name="eurusd_xgboost",
    val_ratio=0.15,
    test_ratio=0.15,
    cross_validate=True,
    n_cv_splits=5,
)
```

### Model Training

There is no CLI entry point — the training pipeline is used as a **Python API**. Create a training script (or run in a notebook/Jupyter session):

```python
import numpy as np
from kael_trading_bot.config import IngestionConfig
from kael_trading_bot.ingestion import ForexDataFetcher
from kael_trading_bot.features.pipeline import build_feature_matrix, FeatureConfig
from src.kael_trading_bot.training.pipeline import PipelineConfig, TrainingPipeline

# 1. Fetch data
ingestion_cfg = IngestionConfig(pairs=("EURUSD=X",))
fetcher = ForexDataFetcher(ingestion_cfg)
df = fetcher.get("EURUSD=X")

# 2. Build features
feature_cfg = FeatureConfig()
features_df = build_feature_matrix(df, config=feature_cfg)

# 3. Prepare X, y
target_col = "target_direction_1"  # 1-period ahead directional target
feature_cols = [c for c in features_df.columns if c.startswith(("sma_", "ema_", "rsi_", "macd_", "bb_", "atr_", "rolling_", "hour_", "dayofweek_"))]
X = features_df[feature_cols].values
y = features_df[target_col].values

# 4. Train
pipeline_cfg = PipelineConfig(
    model_type="xgboost",
    model_name="eurusd_xgboost",
)
pipeline = TrainingPipeline(pipeline_cfg)
result = pipeline.run(X, y, feature_names=feature_cols)

print(f"Test F1: {result.best_test_f1:.4f}")
print(f"Model saved to: {result.saved_path}")
```

#### What the pipeline does

1. **Data ingestion** — fetches forex OHLCV data via Yahoo Finance and caches it locally as Parquet files in `.cache/forex_data/`
2. **Feature engineering** — computes technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR), rolling window statistics, time-based features, and target labels (future returns and directional signals)
3. **Time-aware splitting** — splits data into train (70%), validation (15%), and test (15%) sets chronologically to prevent future leakage
4. **(Optional) Cross-validation** — performs time-series cross-validation with configurable number of folds
5. **Model training** — trains the selected model type with default or custom hyperparameters
6. **Evaluation** — computes classification metrics (accuracy, precision, recall, F1, ROC-AUC) and trading metrics (hit rate, avg return per trade, Sharpe ratio, max drawdown)
7. **Persistence** — saves the trained model as `model.joblib` with a `metadata.json` sidecar
8. **Logging** — appends a structured JSON-lines record to `logs/training_runs.jsonl`

#### Trained model output

Models are saved to the `models/` directory by default, organised as:

```
models/
└── <model_name>/
    └── <model_version>/
        ├── model.joblib       # Serialised model binary
        └── metadata.json      # Training config, metrics, timestamp
```

The model version defaults to a timestamp string like `v20240115T103000`. Use `ModelPersistence.list_models()` and `ModelPersistence.list_versions(name)` to discover saved models.

#### Available Model Types

| Model Type            | Enum Value              | Key Hyperparameters (defaults)                          |
| --------------------- | ----------------------- | ------------------------------------------------------- |
| XGBoost               | `"xgboost"`             | `n_estimators=200`, `max_depth=6`, `learning_rate=0.05` |
| LightGBM              | `"lightgbm"`            | `n_estimators=200`, `max_depth=6`, `learning_rate=0.05` |
| Random Forest         | `"random_forest"`       | `n_estimators=200`, `max_depth=10`                      |
| Logistic Regression   | `"logistic_regression"` | `max_iter=1000`, `C=1.0`                                |

### Using a Trained Model

Load a previously trained model and its metadata using `ModelPersistence`:

```python
from src.kael_trading_bot.training.persistence import ModelPersistence

persistence = ModelPersistence(directory="models")
model, metadata = persistence.load(
    model_name="eurusd_xgboost",
    model_version="v20240115T103000",  # or use list_versions() to find one
)

print(f"Model type: {metadata.model_type}")
print(f"Trained at: {metadata.trained_at}")
print(f"Test metrics: {metadata.metrics.get('test')}")

# Generate predictions
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]  # probability of positive class
```

> ⚠️ **Disclaimer:** This bot is for educational and research purposes only. Forex trading involves significant risk. Always use a demo/paper trading account and never risk capital you cannot afford to lose.

### Running Tests

The project uses `pytest` for testing. Tests are located in the `tests/` directory:

```bash
# Run all tests
pytest

# Run tests for a specific module
pytest tests/test_training_pipeline.py

# Run with coverage report
pytest --cov=src/kael_trading_bot
```

---

## Project Structure

```
src/kael_trading_bot/
├── __init__.py
├── config.py           # IngestionConfig (forex pairs, dates, caching)
├── ingestion.py        # ForexDataFetcher (Yahoo Finance OHLCV)
├── features/           # Technical feature engineering
│   ├── __init__.py
│   ├── pipeline.py     # build_feature_matrix, FeatureConfig
│   ├── indicators.py   # SMA, EMA, RSI, MACD, BB, ATR
│   ├── temporal.py     # Time-based features, rolling stats
│   └── targets.py      # Future returns, directional targets
└── training/           # ML model training pipeline
    ├── __init__.py
    ├── pipeline.py     # TrainingPipeline, PipelineConfig
    ├── models.py       # ModelRegistry, ModelType (XGBoost, LightGBM, RF, LR)
    ├── splitting.py    # TimeSeriesSplitter (chronological split)
    ├── evaluation.py   # Classification + trading metrics
    ├── persistence.py  # ModelPersistence (save/load with metadata)
    └── logging.py      # TrainingLogger (JSON-lines run history)

tests/                  # Unit tests
models/                 # Saved trained models (generated at runtime)
logs/                   # Training run logs (generated at runtime)
.cache/                 # Cached forex data Parquet files (generated at runtime)
```

---

## License

See [LICENSE](LICENSE) for details.
