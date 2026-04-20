"""FastAPI application factory.

Creates and configures the ASGI application that serves both the REST API
and the HTML pages for the web UI.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from kael_trading_bot.web.routes import router as api_router
from kael_trading_bot.web.templates import render_template

logger = logging.getLogger(__name__)

# Resolve paths relative to this package directory.
_WEB_DIR = Path(__file__).parent
_STATIC_DIR = _WEB_DIR / "static"


def create_app() -> FastAPI:
    """Create and return the configured FastAPI application."""
    app = FastAPI(
        title="Kael Trading Bot",
        description="ML-based forex trading bot — Web UI & API",
        version="0.1.0",
    )

    # Mount static files (CSS, JS) at /static/
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # Register REST API routes
    app.include_router(api_router)

    # ------------------------------------------------------------------
    # Page routes (server-rendered HTML)
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Redirect root to the training page."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/training", status_code=307)

    @app.get("/training", response_class=HTMLResponse)
    async def training_page():
        """Render the model training page."""
        html = render_template("training.html", page="training")
        return HTMLResponse(content=html)

    return app
