# Kael Trading Bot — Technical Overview

## Repository Information

- **Name**: [unitz007/kael-trading-bot](https://github.com/unitz007/kael-trading-bot)

## Tech Stack

- **Backend Logic**: Python
- **Frontend**: JavaScript, HTML, CSS

## Core Directories and Files

- `main.py` — CLI entry point
- `src/` — Core Python modules
- `frontend/` — Web UI components
- `tests/` — Test suite
- `Dockerfile` — Container build definition
- `docker-compose.yml` — Multi-service orchestration

## Containerization Strategy

The project is containerized using Docker:

- **Dockerfile**: Defines the container image for the service
- **docker-compose.yml`: Orchestrates multi-container deployments
  - Exposes backend on port 5000
  - Exposes frontend on port 3000
- **Environment Variables**: Configurable via `.env`

## Primary Features

- **Broker API Integration**: Pulls forex data from Yahoo Finance
- **Market Data Parsing**: Processes OHLCV data to extract key financial indicators
- **Trade Execution**: Generates actionable trade setups based on parsed data
- **Activity Logging**: Tracks operations and stores metadata for auditability