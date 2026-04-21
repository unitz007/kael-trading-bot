"""Telegram notifier — sends trade setup alerts via the Telegram Bot API.

This module provides :class:`TelegramNotifier`, a one-way notification
client that formats :class:`~kael_trading_bot.trade_setup.TradeSetup`
objects into human-readable messages and delivers them to a Telegram
chat.

Configuration is driven entirely by environment variables:

* ``TELEGRAM_ENABLED``      — ``"1"`` / ``"true"`` to enable (default: off).
* ``TELEGRAM_BOT_TOKEN``   — Bot token from ``@BotFather``.
* ``TELEGRAM_CHAT_ID``     — Target chat / channel ID.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Telegram Bot API base URL
_TELEGRAM_API = "https://api.telegram.org"


def _env_bool(key: str, default: bool = False) -> bool:
    """Return ``True`` when the environment variable *key* is truthy."""
    val = os.getenv(key, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off", ""):
        return False
    return default


class TelegramNotifier:
    """One-way Telegram notification client for trade setups.

    Parameters
    ----------
    bot_token:
        Telegram Bot API token.  Defaults to ``TELEGRAM_BOT_TOKEN`` env var.
    chat_id:
        Target chat ID.  Defaults to ``TELEGRAM_CHAT_ID`` env var.
    enabled:
        Whether notifications are active.  Defaults to ``TELEGRAM_ENABLED``
        env var (disabled if unset or falsy).
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        enabled: bool | None = None,
        timeout: int = 10,
    ) -> None:
        self.bot_token: str = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id: str = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled: bool = enabled if enabled is not None else _env_bool("TELEGRAM_ENABLED")
        self.timeout: int = timeout
        self._sent_keys: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify_trade_setup(self, setup: Any) -> bool:
        """Send a Telegram notification for *setup*.

        Parameters
        ----------
        setup:
            A :class:`~kael_trading_bot.trade_setup.TradeSetup` instance (or
            any object with the expected attributes).

        Returns
        -------
        bool
            ``True`` if the message was sent successfully, ``False`` otherwise
            (including when notifications are disabled).
        """
        if not self.enabled:
            logger.debug("Telegram notifications are disabled — skipping.")
            return False

        if not self.bot_token or not self.chat_id:
            logger.warning(
                "Telegram is enabled but bot_token or chat_id is not configured. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables."
            )
            return False

        # Deduplication: same pair + timeframe + generated_at
        key = self._dedup_key(setup)
        if key in self._sent_keys:
            logger.debug("Duplicate trade setup notification skipped: %s", key)
            return False

        # checks if setup has a confidence greater than threshold.
        MAX_CONFIDENCE_THRESHOLD = os.getenv("MAX_CONFIDENCE_THRESHOLD", 80)
        if setup.confidence * 100 >= int(MAX_CONFIDENCE_THRESHOLD):
            message = self._format_message(setup)
            return self._send(message, key)

        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup_key(setup: Any) -> str:
        """Build a deduplication key from setup attributes.

        Uses pair, timeframe, and generated_at to uniquely identify a setup.
        """
        pair = getattr(setup, "pair", "unknown")
        timeframe = getattr(setup, "timeframe", "unknown")
        generated_at = getattr(setup, "generated_at", "")
        return f"{pair}:{timeframe}:{generated_at}"

    @staticmethod
    def _format_message(setup: Any) -> str:
        """Format a trade setup into a Telegram-friendly Markdown message."""
        pair = getattr(setup, "pair", "N/A")
        timeframe = getattr(setup, "timeframe", "N/A")
        direction = getattr(setup, "direction", "N/A").upper()
        entry = getattr(setup, "entry_price", 0)
        sl = getattr(setup, "stop_loss", 0)
        tp = getattr(setup, "take_profit", 0)
        rr = getattr(setup, "rr_ratio", 0)
        confidence = getattr(setup, "confidence", 0)
        generated_at = getattr(setup, "generated_at", "N/A")

        # Map direction to an arrow emoji
        arrow = "🟢" if direction == "BUY" else "🔴" if direction == "SELL" else "⚪"

        text = (
            f"{arrow} *New Trade Setup*\n\n"
            f"📋 *Pair:* {pair}\n"
            f"⏱ *Timeframe:* {timeframe}\n"
            f"📊 *Direction:* {direction}\n"
            f"💰 *Entry:* `{entry:.5f}`\n"
            f"🛑 *Stop Loss:* `{sl:.5f}`\n"
            f"🎯 *Take Profit:* `{tp:.5f}`\n"
            f"📈 *R:R Ratio:* 1:{rr:.2f}\n"
            f"🧠 *Confidence:* {confidence * 100:.1f}%\n"
            f"🕐 *Detected:* {generated_at}\n"
        )
        return text

    def _send(self, text: str, dedup_key: str) -> bool:
        """POST *text* to the Telegram Bot API ``sendMessage`` endpoint.

        Returns ``True`` on success, ``False`` on any failure (logged).
        """
        url = f"{_TELEGRAM_API}/bot{self.bot_token}/sendMessage"
        payload: dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()

            result = response.json()
            if not result.get("ok"):
                logger.warning(
                    "Telegram API returned ok=false: %s",
                    result.get("description", "unknown error"),
                )
                return False

            self._sent_keys.add(dedup_key)
            logger.info("Telegram notification sent (key=%s).", dedup_key)
            return True

        except requests.exceptions.Timeout:
            logger.error("Telegram notification timed out (key=%s).", dedup_key)
            return False
        except requests.exceptions.ConnectionError:
            logger.error(
                "Telegram notification failed — connection error (key=%s).",
                dedup_key,
            )
            return False
        except requests.exceptions.HTTPError as exc:
            logger.error(
                "Telegram notification failed — HTTP %s (key=%s): %s",
                exc.response.status_code if exc.response is not None else "?",
                dedup_key,
                exc,
            )
            return False
        except Exception as exc:  # pragma: no cover — defensive
            logger.exception(
                "Unexpected error sending Telegram notification (key=%s): %s",
                dedup_key,
                exc,
            )
            return False
