"""FastAPI application — serves the Kael Trading Bot web UI.

Endpoints
----------
GET /
    Redirects to the predictions page.

GET /predictions
    Renders the predictions page showing all forex pairs and their
    model availability status.

POST /api/predictions
    Generates a prediction for the given pair and returns JSON.

GET /api/predictions/status
    Returns model availability status for all pairs as JSON.

Static assets are served from ``src/kael_trading_bot/web/static/`` and
templates from ``src/kael_trading_bot/web/templates/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.kael_trading_bot.web.predictions import (
    PredictionService,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_WEB_DIR = Path(__file__).parent
_TEMPLATE_DIR = _WEB_DIR / "templates"
_STATIC_DIR = _WEB_DIR / "static"

# ---------------------------------------------------------------------------
# App & service
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Kael Trading Bot",
    description="ML-based forex trading bot — Web UI",
    version="0.1.0",
)

templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
service = PredictionService()

# Serve static files if the directory exists
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> RedirectResponse:
    """Redirect root to the predictions page."""
    return RedirectResponse(url="/predictions", status_code=307)


@app.get("/predictions", response_class=HTMLResponse)
async def predictions_page(request: Request) -> HTMLResponse:
    """Render the predictions page."""
    statuses = service.get_all_model_statuses()
    pairs = service.list_available_pairs()

    # Build display-friendly pair info
    pair_info = []
    for pair, status in zip(pairs, statuses):
        display_name = pair.replace("=X", "")
        pair_info.append(
            {
                "pair": pair,
                "display_name": display_name,
                "model_available": status.available,
                "model_version": status.latest_version,
                "trained_at": status.trained_at,
            }
        )

    return templates.TemplateResponse(
        "predictions.html",
        {
            "request": request,
            "pair_info": pair_info,
        },
    )


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@app.get("/api/predictions/status")
async def api_prediction_status() -> JSONResponse:
    """Return model availability for all pairs."""
    statuses = service.get_all_model_statuses()
    return JSONResponse(
        content=[
            {
                "pair": s.pair,
                "display_name": s.pair.replace("=X", ""),
                "model_available": s.available,
                "model_version": s.latest_version,
                "trained_at": s.trained_at,
            }
            for s in statuses
        ]
    )


@app.post("/api/predictions")
async def api_predict(request: Request) -> JSONResponse:
    """Generate a prediction for the requested pair.

    Expects JSON body: ``{"pair": "EURUSD=X"}``
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    pair = body.get("pair")
    if not pair:
        raise HTTPException(status_code=400, detail="Missing 'pair' field")

    try:
        result = service.predict(pair)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return JSONResponse(
        content={
            "pair": result.pair,
            "display_name": result.pair.replace("=X", ""),
            "direction": result.direction,
            "confidence": round(result.confidence, 4),
            "predicted_return": (
                round(result.predicted_return, 6)
                if result.predicted_return is not None
                else None
            ),
            "model_version": result.model_version,
            "model_type": result.model_type,
            "trained_at": result.trained_at,
            "generated_at": result.generated_at,
        }
    )
