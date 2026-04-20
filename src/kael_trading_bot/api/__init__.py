"""REST API package for the Kael Trading Bot.

Exposes bot capabilities (forex data, model training, predictions)
as HTTP endpoints via Flask.
"""

from kael_trading_bot.api.app import create_app  # noqa: F401

__all__ = ["create_app"]
