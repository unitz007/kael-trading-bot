# Kael Trading Bot

An ML-based forex trading bot that uses machine learning models to analyze market data, identify trading opportunities, and execute trades on forex markets.

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
pip install -r requirements.txt
```

### 4. Verify the installation

```bash
python -c "import kael_trading_bot; print('Package installed successfully!')"
```

## Project Structure

```
kael-trading-bot/
├── src/
│   └── kael_trading_bot/
│       └── __init__.py       # Package entry point
├── requirements.txt           # Pinned third-party dependencies
├── pyproject.toml             # Project metadata and build config
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## Development

The package is installed in editable mode for development:

```bash
pip install -e .
```

## License

TBD
