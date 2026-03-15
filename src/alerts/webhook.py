"""Alert delivery system for the Contrarian Options Alpha Engine.

Reads channel credentials exclusively from environment variables (keys are
defined in ``config/alerts.yaml``) so no secrets ever appear in source or
config files.  All network I/O is async via ``aiohttp``.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal rate-limit tracker
# ---------------------------------------------------------------------------

@dataclass
class _RateLimitState:
    """Tracks the last send timestamp per event type.

    Attributes:
        last_sent: Maps event_type -> epoch timestamp of last successful send.
        min_interval: Minimum seconds that must elapse between sends of the
            same event type.
    """

    last_sent: dict[str, float] = field(default_factory=dict)
    min_interval: float = 30.0

    def is_allowed(self, event_type: str) -> bool:
        """Return True if enough time has elapsed since the last send.

        Args:
            event_type: The event category being tested.

        Returns:
            True when the event is not rate-limited.
        """
        last = self.last_sent.get(event_type, 0.0)
        return (time.monotonic() - last) >= self.min_interval

    def record(self, event_type: str) -> None:
        """Record a successful send for rate-limit accounting.

        Args:
            event_type: The event category that was sent.
        """
        self.last_sent[event_type] = time.monotonic()


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class AlertManager:
    """Async alert dispatcher supporting Telegram and Discord channels.

    Credentials are never stored in config files; the YAML specifies the
    *names* of environment variables to read at construction time.

    Args:
        config_path: Path to the alerts YAML configuration file.
    """

    def __init__(self, config_path: str = "config/alerts.yaml") -> None:
        self._cfg = self._load_config(config_path)
        alerts_cfg: dict[str, Any] = self._cfg.get("alerts", {})

        # Telegram
        tg_cfg = alerts_cfg.get("telegram", {})
        self._telegram_enabled: bool = tg_cfg.get("enabled", False)
        self._telegram_token: str | None = self._read_env(
            tg_cfg.get("bot_token_env", ""), "Telegram bot token"
        )
        self._telegram_chat_id: str | None = self._read_env(
            tg_cfg.get("chat_id_env", ""), "Telegram chat ID"
        )

        # Discord
        dc_cfg = alerts_cfg.get("discord", {})
        self._discord_enabled: bool = dc_cfg.get("enabled", False)
        self._discord_webhook_url: str | None = self._read_env(
            dc_cfg.get("webhook_url_env", ""), "Discord webhook URL"
        )

        # Enabled event types
        self._events: dict[str, bool] = alerts_cfg.get("events", {})

        # Rate limiting
        rl_cfg = alerts_cfg.get("rate_limit", {})
        self._rate_limit = _RateLimitState(
            min_interval=float(rl_cfg.get("min_interval", 30))
        )

    # ------------------------------------------------------------------
    # Public send interface
    # ------------------------------------------------------------------

    async def send(
        self,
        event_type: str,
        message: str,
        data: dict | None = None,
    ) -> None:
        """Dispatch an alert to all enabled channels.

        Silently skips delivery if the same ``event_type`` was sent within
        ``min_interval`` seconds.

        Args:
            event_type: Logical event category (e.g. ``"fills"``).
            message: Human-readable alert text.
            data: Optional supplementary key/value context (currently unused
                by delivery backends but available for future extensions).
        """
        if not self._rate_limit.is_allowed(event_type):
            logger.debug(
                "Alert rate-limited for event_type=%s — skipping.", event_type
            )
            return

        self._rate_limit.record(event_type)

        if self._telegram_enabled:
            await self._send_telegram(message)

        if self._discord_enabled:
            await self._send_discord(message)

    # ------------------------------------------------------------------
    # Convenience event methods
    # ------------------------------------------------------------------

    async def on_fill(
        self, symbol: str, side: str, qty: int, price: float
    ) -> None:
        """Send a trade-fill alert.

        Args:
            symbol: Option contract symbol.
            side: ``"BUY"`` or ``"SELL"``.
            qty: Number of contracts filled.
            price: Fill price per contract.
        """
        if not self._events.get("fills", True):
            return
        message = (
            f"<b>Trade Fill</b>\n"
            f"Symbol: <code>{symbol}</code>\n"
            f"Side: {side}  Qty: {qty}  Price: ${price:.2f}"
        )
        await self.send("fills", message, {"symbol": symbol, "side": side, "qty": qty, "price": price})

    async def on_risk_warning(self, message: str) -> None:
        """Send a risk management warning alert.

        Args:
            message: Human-readable description of the risk condition.
        """
        if not self._events.get("risk_warnings", True):
            return
        await self.send("risk_warnings", f"<b>Risk Warning</b>\n{message}")

    async def on_circuit_breaker(self, win_rate: float) -> None:
        """Send a circuit-breaker triggered alert.

        Args:
            win_rate: Current rolling win rate that triggered the breaker.
        """
        if not self._events.get("circuit_breaker", True):
            return
        message = (
            f"<b>Circuit Breaker Triggered</b>\n"
            f"Rolling win rate has dropped to {win_rate:.1%}. "
            f"Trading halted pending review."
        )
        await self.send("circuit_breaker", message, {"win_rate": win_rate})

    async def on_daily_summary(self, metrics: dict) -> None:
        """Send the end-of-day performance summary.

        Args:
            metrics: Dictionary produced by
                :meth:`~src.broker.paper_trader.PaperTradingOrchestrator.daily_summary`.
        """
        if not self._events.get("daily_summary", True):
            return
        total = metrics.get("total_trades", 0)
        win_rate = metrics.get("win_rate", 0.0)
        total_pnl = metrics.get("total_pnl", 0.0)
        capital = metrics.get("capital", 0.0)
        message = (
            f"<b>Daily Summary</b>\n"
            f"Trades: {total}  Win Rate: {win_rate:.1%}\n"
            f"P&amp;L: ${total_pnl:+.2f}  Capital: ${capital:.2f}"
        )
        await self.send("daily_summary", message, metrics)

    async def on_error(self, error: str) -> None:
        """Send an error/exception alert.

        Args:
            error: Short description of the error that occurred.
        """
        if not self._events.get("errors", True):
            return
        await self.send("errors", f"<b>Error</b>\n<code>{error}</code>")

    # ------------------------------------------------------------------
    # Channel delivery backends
    # ------------------------------------------------------------------

    async def _send_telegram(self, message: str) -> None:
        """POST a message to the configured Telegram chat.

        Args:
            message: HTML-formatted message body.
        """
        if not self._telegram_token or not self._telegram_chat_id:
            logger.warning(
                "Telegram credentials missing — cannot deliver alert."
            )
            return

        try:
            import aiohttp  # local import to avoid hard dependency at module load

            url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
            payload = {
                "chat_id": self._telegram_chat_id,
                "text": message,
                "parse_mode": "HTML",
            }
            async with aiohttp.ClientSession() as session, session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(
                        "Telegram delivery failed (HTTP %d): %s", resp.status, body
                    )
                else:
                    logger.debug("Telegram alert delivered.")
        except Exception as exc:
            logger.error("Telegram send error: %s", exc)

    async def _send_discord(self, message: str) -> None:
        """POST a message embed to the configured Discord webhook.

        Args:
            message: Plain or lightly-formatted message body (HTML tags are
                stripped for Discord compatibility).
        """
        if not self._discord_webhook_url:
            logger.warning(
                "Discord webhook URL missing — cannot deliver alert."
            )
            return

        # Discord doesn't render Telegram-style HTML; strip basic tags.
        clean = (
            message.replace("<b>", "**")
            .replace("</b>", "**")
            .replace("<code>", "`")
            .replace("</code>", "`")
            .replace("<br>", "\n")
            .replace("&amp;", "&")
        )

        try:
            import aiohttp

            payload = {
                "embeds": [
                    {
                        "description": clean,
                        "color": 0x2ECC71,
                    }
                ]
            }
            async with aiohttp.ClientSession() as session, session.post(
                self._discord_webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status not in (200, 204):
                    body = await resp.text()
                    logger.warning(
                        "Discord delivery failed (HTTP %d): %s", resp.status, body
                    )
                else:
                    logger.debug("Discord alert delivered.")
        except Exception as exc:
            logger.error("Discord send error: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_env(env_key: str, description: str) -> str | None:
        """Read a value from an environment variable.

        Args:
            env_key: Name of the environment variable to read.
            description: Human-readable label used in warning messages.

        Returns:
            The variable's value, or ``None`` if it is unset or empty.
        """
        if not env_key:
            return None
        value = os.environ.get(env_key, "").strip()
        if not value:
            logger.warning(
                "%s not set (env var: %s). Channel may be non-functional.",
                description,
                env_key,
            )
            return None
        return value

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """Load and return the YAML alert configuration.

        Args:
            config_path: Filesystem path to the YAML file.

        Returns:
            Parsed configuration dict, or an empty dict on failure.
        """
        path = Path(config_path)
        if not path.exists():
            logger.warning(
                "Alert config not found at %s. Using empty defaults.", config_path
            )
            return {}
        try:
            with path.open() as fh:
                return yaml.safe_load(fh) or {}
        except Exception as exc:
            logger.error("Failed to load alert config from %s: %s", config_path, exc)
            return {}
