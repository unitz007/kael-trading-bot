# DOCKER.md

# Running Kael Trading Bot with Docker

This guide explains how to build and run the Kael Trading Bot using Docker and Docker Compose.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (version 20.10 or later)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2 or later, included with Docker Desktop)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/unitz007/kael-trading-bot.git
cd kael-trading-bot

# Build and start both services
docker compose up --build
```

The application will be available at:

| Service  | URL                       | Description                          |
| -------- | ------------------------- | ------------------------------------ |
| Frontend | http://localhost:3000     | React web UI (served by Nginx)       |
| Backend  | http://localhost:5000     | FastAPI REST API + docs              |

Open **http://localhost:3000** in your browser to access the web UI. API requests from the frontend are proxied to the backend automatically via Nginx.

## Running in Detached Mode

```bash
# Start in the background
docker compose up -d --build

# View logs
docker compose logs -f

# Stop
docker compose down
```

## Configuration via Environment Variables

All runtime configuration is driven by environment variables. You can set them in a `.env` file at the repository root (alongside `docker-compose.yml`) or pass them directly.

### Backend Variables

| Variable             | Description                           | Default             |
| -------------------- | ------------------------------------- | ------------------- |
| `KAEL_PORT`          | Port the API server listens on        | `5000`              |
| `KAEL_START_DATE`    | Data start date (ISO 8601, inclusive) | `2020-01-01`        |
| `KAEL_END_DATE`      | Data end date (ISO 8601, inclusive)   | `2025-01-01`        |
| `KAEL_INTERVAL`      | Data frequency (`1d`, `1h`, etc.)     | `1d`                |
| `KAEL_CACHE_DIR`     | Directory for cached Parquet data     | `.cache/forex_data` |
| `PYTHONUNBUFFERED`   | Ensure Python output is not buffered  | `1`                 |

### Frontend Variables

| Variable         | Description                          | Default |
| ---------------- | ------------------------------------ | ------- |
| `FRONTEND_PORT`  | Host port mapped to Nginx container  | `3000`  |

### Example `.env` file

```env
# docker-compose will automatically read this file
KAEL_PORT=5000
FRONTEND_PORT=3000
KAEL_START_DATE=2022-01-01
KAEL_END_DATE=2025-06-01
KAEL_INTERVAL=1d
```

## Persistent Data

Trained models, cached forex data, and training logs are stored in Docker volumes so they survive container restarts:

| Volume           | Mount path in container | Purpose                |
| ---------------- | ----------------------- | ---------------------- |
| `kael-model-data` | `/app/models`           | Trained model files    |
| `kael-cache-data` | `/app/.cache`           | Cached Parquet data    |
| `kael-log-data`   | `/app/logs`             | Training run logs      |

To remove volumes (e.g., for a clean start):

```bash
docker compose down -v
```

## Building Images Individually

If you only need to build one service:

```bash
# Backend only
docker build -t kael-bot-backend .

# Frontend only
docker build -t kael-bot-frontend ./frontend
```

## API Documentation

When the backend is running, FastAPI's interactive docs are available at:

- Swagger UI: http://localhost:5000/docs
- ReDoc: http://localhost:5000/redoc

## Troubleshooting

**Frontend shows "Backend unavailable"**

- Verify the backend container is running: `docker compose ps`
- Check backend logs: `docker compose logs backend`
- Ensure both services are on the same Docker network (automatic with compose)

**Training takes too long / times out**

- The Nginx proxy is configured with a 300s timeout for API requests to accommodate training. If training exceeds this, increase `proxy_read_timeout` in `frontend/nginx.conf`.

**Port already in use**

- Change the host port mapping in `.env` or docker-compose.yml:
  ```env
  KAEL_PORT=8080
  FRONTEND_PORT=8081
  ```

## Development Without Docker

For local development, you can run the application outside of Docker:

1. Install Python 3.11 or higher
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   If `pip` is not found, try using:
   ```bash
   python -m pip install -r requirements.txt
   ```
4. Install the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
   
   Or if `pip` is not found:
   ```bash
   python -m pip install -e ".[dev]"
   ```
5. Configure your environment variables (optionally using a `.env` file)
6. Start the development server:
   ```bash
   python main.py serve
   ```

This approach allows you to develop without Docker while still maintaining compatibility with the containerized production environment.