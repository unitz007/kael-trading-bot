# Kael Trading Bot – ML-based Forex Trading Bot

Kael Trading Bot is a Python-based framework that ingests real‑time forex data,\nesengineers technical indicators, trains supervised classifiers, and exposes\npredictions via a lightweight REST API and web UI.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contribution Guidelines](#contribution-guidelines)

## Installation

### pip (recommended)

```bash
# clone
git clone https://github.com/unitz007/kael-trading-bot.git
cd kael-trading-bot

# create & activate venv (recommended)
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

### Docker Compose

```bash
docker compose up --build
# Frontend UI will be served on http://localhost:3000
# Backend/REST API will listen on http://localhost:5000
```

### Environment

Set the following environment variables (or create a `.env` file):

| Variable | Description | Default |
|----------|-------------|---------|
| `KAEL_START_DATE` | Data start date (ISO) | `2020-01-01` |
| `KAEL_END_DATE` | Data end date (ISO) | `2025-01-01` |
| `KAEL_INTERVAL` | Data frequency (`1d`, `1h`, …) | `1d` |
| `KAEL_CACHE_DIR` | Cached Pearson data dir | `.cache/forex_data` |

## Usage

Start the API and UI:

```bash
python main.py serve
```

Alternatively, with uvicorn directly:

```bash
uvicorn kael_trading_bot.api:create_app --factory --host 0.0.0.0 --port 5000
```

Trigger a training job for a pair (e.g. `EURUSD=X`):

```bash
curl -X POST http://localhost:5000/api/v1/pairs/EURUSD%3DX/train
```

Prediction:

```bash
curl http://localhost:5000/api/v1/pairs/EURUSD%3DX/predict
```

## Contribution Guidelines

1. **Fork** the repository and clone your fork.
2. Create a branch following the convention: `feat/<short-description>` or `fix/<short-description>`.
3. Make your changes, run `ruff check` and `mypy` locally to satisfy style.
4. Add tests for new behavior.
5. Open a pull request. The CI will run `pytest`, `ruff`, `mypy`, and `build`. If all checks pass, a reviewer will merge.

### Repository Conventions

- Python source lives in `src/kael_trading_bot`.
- Tests are under `tests/`.
- Documentation is in `README.md` only; external docs are out of scope.
- Models are persisted in `models/`.
- Docker build is defined by `Dockerfile` and `docker-compose.yml`.

## License

MIT. See [LICENSE](LICENSE) for details.