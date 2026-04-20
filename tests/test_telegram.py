"""Tests for the Telegram notification module."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import requests

from kael_trading_bot.telegram.notifier import TelegramNotifier, _env_bool


# ---------------------------------------------------------------------------
# Lightweight stand-in for a TradeSetup
# ---------------------------------------------------------------------------


@dataclass
class _FakeTradeSetup:
    pair: str = "EURUSD=X"
    timeframe: str = "1h"
    direction: str = "buy"
    entry_price: float = 1.08500
    stop_loss: float = 1.08200
    take_profit: float = 1.09100
    confidence: float = 0.78
    rr_ratio: float = 2.0
    generated_at: str = "2025-01-15T12:00:00+00:00"


# ---------------------------------------------------------------------------
# _env_bool helper
# ---------------------------------------------------------------------------


class TestEnvBool:
    def test_true_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for val in ("1", "true", "True", "TRUE", "yes", "YES", "on", "ON"):
            monkeypatch.setenv("_TEST_VAR", val)
            assert _env_bool("_TEST_VAR") is True

    def test_false_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for val in ("0", "false", "False", "FALSE", "no", "NO", "off", "OFF", ""):
            monkeypatch.setenv("_TEST_VAR", val)
            assert _env_bool("_TEST_VAR") is False

    def test_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("_TEST_VAR", raising=False)
        assert _env_bool("_TEST_VAR") is False

    def test_custom_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("_TEST_VAR", raising=False)
        assert _env_bool("_TEST_VAR", default=True) is True


# ---------------------------------------------------------------------------
# TelegramNotifier — construction
# ---------------------------------------------------------------------------


class TestNotifierConstruction:
    def test_defaults_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TELEGRAM_ENABLED", raising=False)
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        notifier = TelegramNotifier()
        assert notifier.enabled is False
        assert notifier.bot_token == ""
        assert notifier.chat_id == ""

    def test_explicit_args_override_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "env-token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "env-chat")
        monkeypatch.setenv("TELEGRAM_ENABLED", "0")
        notifier = TelegramNotifier(
            bot_token="arg-token",
            chat_id="arg-chat",
            enabled=True,
        )
        assert notifier.bot_token == "arg-token"
        assert notifier.chat_id == "arg-chat"
        assert notifier.enabled is True

    def test_env_vars_respected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "456")
        monkeypatch.setenv("TELEGRAM_ENABLED", "1")
        notifier = TelegramNotifier()
        assert notifier.bot_token == "tok123"
        assert notifier.chat_id == "456"
        assert notifier.enabled is True


# ---------------------------------------------------------------------------
# TelegramNotifier — disabled / missing config
# ---------------------------------------------------------------------------


class TestNotifierDisabled:
    def test_notify_returns_false_when_disabled(self) -> None:
        notifier = TelegramNotifier(enabled=False, bot_token="tok", chat_id="123")
        setup = _FakeTradeSetup()
        assert notifier.notify_trade_setup(setup) is False

    def test_notify_returns_false_when_no_token(self) -> None:
        notifier = TelegramNotifier(enabled=True, bot_token="", chat_id="123")
        assert notifier.notify_trade_setup(_FakeTradeSetup()) is False

    def test_notify_returns_false_when_no_chat_id(self) -> None:
        notifier = TelegramNotifier(enabled=True, bot_token="tok", chat_id="")
        assert notifier.notify_trade_setup(_FakeTradeSetup()) is False


# ---------------------------------------------------------------------------
# TelegramNotifier — formatting
# ---------------------------------------------------------------------------


class TestFormatMessage:
    def test_buy_setup_format(self) -> None:
        setup = _FakeTradeSetup(direction="buy")
        msg = TelegramNotifier._format_message(setup)
        assert "🟢" in msg
        assert "BUY" in msg
        assert "EURUSD=X" in msg
        assert "1h" in msg
        assert "1.08500" in msg
        assert "1.08200" in msg
        assert "1.09100" in msg
        assert "1:2.00" in msg
        assert "78.0%" in msg
        assert "2025-01-15T12:00:00+00:00" in msg

    def test_sell_setup_format(self) -> None:
        setup = _FakeTradeSetup(direction="sell")
        msg = TelegramNotifier._format_message(setup)
        assert "🔴" in msg
        assert "SELL" in msg

    def test_unknown_direction(self) -> None:
        setup = _FakeTradeSetup(direction="hold")
        msg = TelegramNotifier._format_message(setup)
        assert "⚪" in msg
        assert "HOLD" in msg


# ---------------------------------------------------------------------------
# TelegramNotifier — deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_same_setup_not_sent_twice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TELEGRAM_ENABLED", "1")
        notifier = TelegramNotifier(
            bot_token="tok", chat_id="123", enabled=True
        )
        setup = _FakeTradeSetup()

        with patch.object(notifier, "_send", return_value=True) as mock_send:
            assert notifier.notify_trade_setup(setup) is True
            assert mock_send.call_count == 1

            # Second call — same key — should be skipped
            assert notifier.notify_trade_setup(setup) is False
            assert mock_send.call_count == 1

    def test_different_timestamps_both_sent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        notifier = TelegramNotifier(
            bot_token="tok", chat_id="123", enabled=True
        )
        setup1 = _FakeTradeSetup(generated_at="2025-01-15T12:00:00+00:00")
        setup2 = _FakeTradeSetup(generated_at="2025-01-15T13:00:00+00:00")

        with patch.object(notifier, "_send", return_value=True) as mock_send:
            assert notifier.notify_trade_setup(setup1) is True
            assert notifier.notify_trade_setup(setup2) is True
            assert mock_send.call_count == 2


# ---------------------------------------------------------------------------
# TelegramNotifier — _send (HTTP interactions)
# ---------------------------------------------------------------------------


class TestSend:
    def _make_notifier(self) -> TelegramNotifier:
        return TelegramNotifier(bot_token="test-token", chat_id="12345", enabled=True)

    @patch("kael_trading_bot.telegram.notifier.requests.post")
    def test_successful_send(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        notifier = self._make_notifier()
        setup = _FakeTradeSetup()

        assert notifier.notify_trade_setup(setup) is True
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "test-token" in call_args[0][0]  # URL contains token
        payload = call_args[1]["json"]
        assert payload["chat_id"] == "12345"
        assert payload["parse_mode"] == "Markdown"
        assert "EURUSD=X" in payload["text"]

    @patch("kael_trading_bot.telegram.notifier.requests.post")
    def test_api_returns_ok_false(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": False, "description": "Bad token"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        notifier = self._make_notifier()
        assert notifier.notify_trade_setup(_FakeTradeSetup()) is False

    @patch("kael_trading_bot.telegram.notifier.requests.post")
    def test_timeout_error(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = requests.exceptions.Timeout("timed out")

        notifier = self._make_notifier()
        assert notifier.notify_trade_setup(_FakeTradeSetup()) is False

    @patch("kael_trading_bot.telegram.notifier.requests.post")
    def test_connection_error(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = requests.exceptions.ConnectionError("conn refused")

        notifier = self._make_notifier()
        assert notifier.notify_trade_setup(_FakeTradeSetup()) is False

    @patch("kael_trading_bot.telegram.notifier.requests.post")
    def test_http_error(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_post.return_value = mock_response
        mock_post.return_value.raise_for_status.side_effect = (
            requests.exceptions.HTTPError(response=mock_response)
        )

        notifier = self._make_notifier()
        assert notifier.notify_trade_setup(_FakeTradeSetup()) is False

    @patch("kael_trading_bot.telegram.notifier.requests.post")
    def test_generic_exception(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = RuntimeError("unexpected")

        notifier = self._make_notifier()
        assert notifier.notify_trade_setup(_FakeTradeSetup()) is False


# ---------------------------------------------------------------------------
# TelegramNotifier — dedup_key helper
# ---------------------------------------------------------------------------


class TestDedupKey:
    def test_with_all_attributes(self) -> None:
        setup = _FakeTradeSetup()
        key = TelegramNotifier._dedup_key(setup)
        assert key == "EURUSD=X:1h:2025-01-15T12:00:00+00:00"

    def test_with_missing_attributes(self) -> None:
        obj = object()
        key = TelegramNotifier._dedup_key(obj)
        assert key == "unknown:unknown:"