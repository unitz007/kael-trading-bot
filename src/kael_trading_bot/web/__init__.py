from __future__ import annotations


"""Kael Trading Bot — Web UI and REST API.

Provides a FastAPI application that serves:
- REST API endpoints for model training management
- Server-rendered HTML pages with vanilla JS for interactivity
"""

from kael_trading_bot.web.app import create_app

__all__ = ["create_app"]
"""Flask web application for the Kael Trading Bot.

Creates and configures the Flask app with blueprint registration.
Run with: ``python -m kael_trading_bot.web``
"""


from flask import Flask

from kael_trading_bot.web.routes import main_bp


def create_app() -> Flask:
    """Application factory: build and configure the Flask app."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    app.register_blueprint(main_bp)

    return app
