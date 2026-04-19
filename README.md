# Kael Trading Bot

ML-based forex trading bot with technical feature engineering and model training pipeline.

---

## Table of Contents

- [Overview](#overview)
- [Usage](#usage)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Training the ML Model](#training-the-ml-model)
  - [Running the Trading Bot](#running-the-trading-bot)
  - [CLI Arguments & Configuration Options](#cli-arguments--configuration-options)

---

## Overview

Kael Trading Bot ingests forex pair data, engineers technical features (indicators, temporal features, targets), trains ML models (XGBoost, LightGBM, scikit-learn), evaluates them with classification and trading-oriented metrics, and persists trained models for use in a live trading workflow.

---

## Usage

### Prerequisites

| Requirement        | Details                                                    |
| ------------------ | ---------------------------------------------------------- |
| **Python**         | 3.10 or higher                                             |
| **Operating system** | Linux, macOS, or Windows (WSL recommended on Windows)     |
| **Data source**    | Internet access to [Yahoo Finance](https://finance.yahoo.com) via the `yfinance` library — no API key required |
| **External tools** | Git (for cloning the repository)                           |

> **Note:** Yahoo Finance provides free forex data without authentication. If you switch to a paid data provider in the future (e.g., Alpha Vantage, OANDA), you will need an API key and must update the data ingestion configuration accordingly.

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
   conda create -n kael-bot python=3.10
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
   - `joblib` — model persistence
   - `matplotlib`, `seaborn` — visualisation
   - `python-dotenv` — environment variable loading

4. **(Optional) Install the package in editable mode for development:**

   ```bash
   pip install -e .
   ```

### Configuration

The project uses a configuration file (`.yaml`) and optionally `.env` for environment variables.

1. **Copy the example configuration** (if provided) or create one:

   The main config file is typically located at the project root. It controls:
   - Forex pairs and date ranges for data ingestion
   - Feature engineering parameters (indicator windows, target horizons)
   - Model hyperparameters and type selection
   - Train/validation/test split ratios
   - Output directories for models and logs

   ```yaml
   # Example configuration keys (refer to the actual config file for the full schema)
   data:
     pairs: ["EURUSD=X"]
     start_date: "2020-01-01"
     end_date: "2024-01-01"
   features:
     # Feature engineering settings
   training:
     model_type: "xgboost"   # Options: xgboost, lightgbm, random_forest
   ```

2. **(Optional) Create a `.env` file** for sensitive or environment-specific settings:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` to set any required environment variables. At minimum this may include:
   - `PYTHONPATH` — ensure the `src` directory is on the Python path
   - Any API keys if you switch from Yahoo Finance to a paid provider

### Training the ML Model

1. **Ensure your configuration file** specifies the forex pairs, date range, and model type you want to train.

2. **Run the training pipeline:**

   ```bash
   python -m src.kael_trading_bot.training.pipeline --config config.yaml
   ```

   This executes the full end-to-end pipeline:
   - **Data ingestion** — fetches forex OHLCV data via Yahoo Finance and caches it locally as Parquet
   - **Feature engineering** — computes technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.), temporal features, and target labels using `FeatureConfig`
   - **Time-aware splitting** — splits data into train, validation, and test sets chronologically (no future leakage)
   - **Model training** — trains the selected model (XGBoost, LightGBM, or Random Forest) using the registry in `training/models`
   - **Evaluation** — computes classification metrics (accuracy, precision, recall, F1, ROC-AUC) and trading metrics (hit rate, avg return per trade, Sharpe ratio, max drawdown) on validation and test sets
   - **Persistence** — saves the trained model along with metadata (hyperparameters, metrics, timestamp) via `joblib`

3. **Trained model output:**

   Trained models are saved to the configured output directory (default: `models/`). Each saved model includes:
   - The serialised model file (`.joblib` or `.pkl`)
   - A metadata JSON file with training configuration and evaluation results

4. **Training logs:**

   Structured training logs are written to the configured log directory, recording each training run's parameters and results for reproducibility.

### Running the Trading Bot

After training and saving a model, you can run the bot to generate trading signals on live or latest data:

```bash
python -m src.kael_trading_bot.bot --config config.yaml --model models/<model_name>.joblib
```

> ⚠️ **Disclaimer:** This bot is for educational and research purposes only. Forex trading involves significant risk. Always use a demo/paper trading account and never risk capital you cannot afford to lose.

#### What the bot does

1. Loads the trained model and its associated metadata
2. Fetches the latest forex data for the configured pairs
3. Engineers the same features used during training
4. Generates directional signals (long/short/neutral) using the model predictions
5. Outputs signals with confidence scores for downstream execution

### CLI Arguments & Configuration Options

| Argument / Option | Description                                      | Example                            |
| ----------------- | ------------------------------------------------ | ---------------------------------- |
| `--config`        | Path to the YAML configuration file              | `--config config.yaml`             |
| `--model`         | Path to a trained model file (`.joblib`)         | `--model models/xgb_eurusd.joblib` |
| `--pairs`         | Forex pairs to process (overrides config)        | `--pairs EURUSD=X GBPUSD=X`        |
| `--start-date`    | Data start date (YYYY-MM-DD)                     | `--start-date 2023-01-01`          |
| `--end-date`      | Data end date (YYYY-MM-DD)                       | `--end-date 2024-01-01`            |
| `--model-type`    | Model type to train (overrides config)           | `--model-type lightgbm`            |
| `--output-dir`    | Directory for output files                       | `--output-dir ./results`           |
| `--log-level`     | Logging verbosity (`DEBUG`, `INFO`, `WARNING`)   | `--log-level DEBUG`                |

#### Available Model Types

| Model Type         | Key in Config | Description                                  |
| ------------------ | ------------- | -------------------------------------------- |
| XGBoost            | `xgboost`     | Gradient boosted trees (default)             |
| LightGBM           | `lightgbm`    | Light gradient boosting, fast training       |
| Random Forest      | `random_forest` | Ensemble of decision trees (scikit-learn) |

#### Environment Variables

| Variable      | Description                                           | Default       |
| ------------- | ----------------------------------------------------- | ------------- |
| `PYTHONPATH`  | Should include `src/` for package imports             | —             |
| `KAEL_CONFIG` | Path to the default configuration file                | `config.yaml` |
| `KAEL_LOG_LEVEL` | Override log level without CLI flag                | `INFO`        |

---

## Project Structure

```
src/kael_trading_bot/
├── __init__.py
├── config/          # Project configuration loading
├── features/        # Technical feature engineering pipeline
├── ingestion/       # Forex data ingestion (Yahoo Finance)
├── training/        # ML model training, evaluation, persistence
└── bot/             # Trading bot execution and signal generation
```

---

## License

See [LICENSE](LICENSE) for details.
