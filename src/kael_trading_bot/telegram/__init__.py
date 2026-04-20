"""Telegram notification integration for the Kael Trading Bot.

Provides a one-way notification channel that sends trade setup alerts
to a configured Telegram chat via the Bot API.
"""

from kael_trading_bot.telegram.notifier import TelegramNotifier

__all__ = ["TelegramNotifier"]
