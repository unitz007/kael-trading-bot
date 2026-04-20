"""Entry point for running the Flask web server.

Usage::

    python -m kael_trading_bot.web
"""

from __future__ import annotations

from kael_trading_bot.web import create_app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
