"""Kael Trading Bot — Web UI and REST API.

Provides a FastAPI application that serves:
- REST API endpoints for model training management
- Server-rendered HTML pages with vanilla JS for interactivity
"""

from kael_trading_bot.web.app import create_app

__all__ = ["create_app"]
