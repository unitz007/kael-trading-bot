# Kael Trading Bot - Technical Summary

## Overview

Kael Trading Bot is a machine learning-based forex trading system that ingests market data, engineers technical features, trains ML models, and provides predictions and trading setups via a REST API and web interface.

## Architecture

The system follows a modular architecture with clearly separated concerns:

### Data Layer
- **Data Source**: Yahoo Finance via the `yfinance` library
- **Ingestion**: Handles fetching and caching of OHLCV (Open, High, Low, Close, Volume) data
- **Caching**: Uses Parquet format for efficient local storage of historical data

### Feature Engineering Layer
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR
- **Temporal Features**: Time-based features (hour, day of week, etc.)
- **Rolling Statistics**: Various window-based statistical measures
- **Target Generation**: Future returns and directional targets for supervised learning

### Machine Learning Layer
- **Supported Models**: 
  - XGBoost
  - LightGBM
  - Random Forest
  - Logistic Regression
- **Training Pipeline**:
  - Chronological data splitting to prevent lookahead bias
  - Cross-validation support
  - Comprehensive evaluation metrics (classification + trading-specific)
  - Model persistence with metadata
- **Prediction Pipeline**: Loads trained models and generates directional predictions

### API Layer
- **Framework**: FastAPI for REST API implementation
- **Endpoints**: 
  - Forex pair data retrieval
  - Model training triggers
  - Prediction results
  - Trade setup generation
  - Model management
- **Real-time Features**: WebSocket support for live updates

### Presentation Layer
- **Web Interface**: React-based frontend served by Nginx
- **Visualization**: Charting capabilities for market data and predictions
- **User Experience**: Intuitive interface for monitoring and interacting with the bot

## Key Components

### Core Modules

1. **Ingestion (`src/kael_trading_bot/ingestion.py`)**
   - Fetches forex data from Yahoo Finance
   - Implements caching mechanism using Parquet files

2. **Feature Engineering (`src/kael_trading_bot/features/`)**
   - `pipeline.py`: Orchestrates feature creation process
   - `indicators.py`: Implements technical indicators
   - `temporal.py`: Time-based feature generation
   - `targets.py`: Creates target variables for training

3. **Training (`src/kael_trading_bot/training/`)**
   - `pipeline.py`: Main training workflow
   - `models.py`: Model registry and instantiation
   - `evaluation.py`: Performance metrics calculation
   - `persistence.py`: Model saving/loading with metadata
   - `splitting.py`: Time series-aware data splitting

4. **API (`src/kael_trading_bot/api/`)**
   - `app.py`: FastAPI application with all endpoints
   - WebSocket support for real-time updates

5. **Trading Logic (`src/kael_trading_bot/trade_setup/`)**
   - Generates actionable trade setups (entry/stop/take-profit)
   - Backtesting capabilities

### Entry Points

1. **CLI (`main.py`)**
   - Commands for training, prediction, and serving
   - Direct programmatic access to bot functionality

2. **REST API**
   - Accessible via HTTP endpoints
   - Web UI integration

## Deployment

### Docker Support
- **Multi-container Architecture**: Separates backend (Python) and frontend (React/Nginx)
- **Persistent Volumes**: For models, cached data, and logs
- **Environment Configuration**: Via .env files or docker-compose overrides

### Requirements
Main dependencies include:
- Data handling: numpy, pandas, pyarrow
- ML frameworks: scikit-learn, xgboost, lightgbm
- Data sources: yfinance
- Technical analysis: ta
- Visualization: matplotlib, seaborn
- Web framework: fastapi, uvicorn
- Persistence: joblib

## Testing

The project includes a comprehensive test suite covering:
- Unit tests for individual components
- Integration tests for pipelines
- Model behavior verification
- API endpoint testing

## Data Flow

1. **Data Ingestion**: Fetch OHLCV data from Yahoo Finance
2. **Feature Engineering**: Transform raw data into predictive features
3. **Model Training**: Train ML models on historical data
4. **Prediction**: Generate trading signals from latest data
5. **Trade Setup**: Convert predictions into actionable trades
6. **Execution**: (External) Execute trades based on setups

## Configuration

The system is configured through:
- Python dataclasses for structured configuration
- Environment variables for runtime settings
- Programmatic overrides for custom behavior

## Persistence

- **Models**: Stored in `models/` directory with versioning
- **Cached Data**: Parquet files in `.cache/` directory
- **Training Logs**: JSON-lines format for analytics