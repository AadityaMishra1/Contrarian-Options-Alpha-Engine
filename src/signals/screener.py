"""Options screener using Polygon.io REST API.

Scans for contrarian options candidates by filtering most-active tickers on
market cap, volume, and RSI using Wilder's exponential smoothing — identical
to the method used in backtest/replay_engine.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import date, timedelta
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_POLYGON_BASE = "https://api.polygon.io"
_SNAPSHOT_ENDPOINT = "/v2/snapshot/locale/us/markets/stocks/tickers"
_AGGS_ENDPOINT = "/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"


# ---------------------------------------------------------------------------
# RSI helper — Wilder's method matching backtest/replay_engine.py
# ---------------------------------------------------------------------------


def _compute_rsi(closes: list[float], period: int = 14) -> float | None:
    """Compute the most recent RSI value using Wilder's exponential smoothing.

    Args:
        closes: Ordered list of closing prices, oldest first.
        period: Look-back period (default 14).

    Returns:
        RSI value in [0, 100], or None if there are insufficient bars.
    """
    if len(closes) < period + 1:
        return None

    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(0.0, delta))
        losses.append(max(0.0, -delta))

    alpha = 1.0 / period

    # Seed with simple average over the first `period` bars
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    # Apply Wilder's smoothing for the remaining bars
    for g, loss in zip(gains[period:], losses[period:], strict=False):
        avg_gain = alpha * g + (1 - alpha) * avg_gain
        avg_loss = alpha * loss + (1 - alpha) * avg_loss

    if avg_loss == 0.0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ---------------------------------------------------------------------------
# OptionsScreener
# ---------------------------------------------------------------------------


class OptionsScreener:
    """Scans for contrarian options candidates using Polygon.io.

    Fetches the most active US equity snapshots, applies market-cap and
    volume filters, then computes RSI from recent daily bars to surface
    oversold candidates suitable for the contrarian strategy.

    Args:
        api_key: Polygon.io API key. Falls back to the ``POLYGON_API_KEY``
            environment variable when not provided.
        min_market_cap: Minimum market capitalisation in dollars.
        min_volume: Minimum daily volume (shares).
        rsi_threshold: Candidates with RSI below this value are returned.
        rsi_period: Period for Wilder's RSI calculation.
        bars_lookback: Number of daily bars fetched per ticker for RSI.
    """

    def __init__(
        self,
        api_key: str | None = None,
        min_market_cap: float = 1e9,
        min_volume: int = 1_000_000,
        rsi_threshold: float = 35.0,
        rsi_period: int = 14,
        bars_lookback: int = 20,
    ) -> None:
        resolved_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Polygon API key is required. Pass api_key= or set the "
                "POLYGON_API_KEY environment variable."
            )
        self._api_key = resolved_key
        self.min_market_cap = min_market_cap
        self.min_volume = min_volume
        self.rsi_threshold = rsi_threshold
        self.rsi_period = rsi_period
        self.bars_lookback = bars_lookback
        self._semaphore = asyncio.Semaphore(5)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def scan(self) -> list[dict[str, Any]]:
        """Scan for oversold candidates across most-active US stocks.

        Returns:
            List of candidate dicts with keys: symbol, price, volume,
            rsi, market_cap.
        """
        async with aiohttp.ClientSession() as session:
            snapshots = await self._fetch_snapshots(session)
            logger.info("Fetched %d ticker snapshots from Polygon", len(snapshots))

            pre_filtered = [
                t for t in snapshots
                if (
                    t.get("market_cap", 0) >= self.min_market_cap
                    and t.get("day", {}).get("v", 0) >= self.min_volume
                )
            ]
            logger.info(
                "%d tickers passed market-cap / volume filter", len(pre_filtered)
            )

            tasks = [
                self._evaluate_ticker(session, t) for t in pre_filtered
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates: list[dict[str, Any]] = []
        for item in results:
            if isinstance(item, Exception):
                logger.debug("Ticker evaluation raised an exception: %s", item)
                continue
            if item is not None:
                candidates.append(item)

        logger.info("Scan complete — %d oversold candidates found", len(candidates))
        return candidates

    async def get_bars(
        self, symbol: str, days: int = 20
    ) -> list[dict[str, Any]]:
        """Fetch recent daily OHLCV bars for a single ticker.

        Args:
            symbol: Equity ticker symbol (e.g. ``"AAPL"``).
            days: Number of calendar days to look back.

        Returns:
            List of bar dicts with keys: t, o, h, l, c, v.
        """
        async with aiohttp.ClientSession() as session:
            return await self._fetch_bars(session, symbol, days)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_snapshots(
        self, session: aiohttp.ClientSession
    ) -> list[dict[str, Any]]:
        """Fetch all ticker snapshots from the Polygon snapshot endpoint."""
        url = f"{_POLYGON_BASE}{_SNAPSHOT_ENDPOINT}"
        params = {"apiKey": self._api_key, "include_otc": "false"}
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("tickers", [])
        except Exception as exc:
            logger.error("Failed to fetch snapshots: %s", exc)
            return []

    async def _fetch_bars(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        days: int,
    ) -> list[dict[str, Any]]:
        """Fetch daily bars for *symbol* over the last *days* calendar days."""
        to_date = date.today().isoformat()
        from_date = (date.today() - timedelta(days=days + 10)).isoformat()
        path = _AGGS_ENDPOINT.format(
            ticker=symbol, from_date=from_date, to_date=to_date
        )
        url = f"{_POLYGON_BASE}{path}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": str(days + 10),
            "apiKey": self._api_key,
        }
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data.get("results", [])

    async def _evaluate_ticker(
        self,
        session: aiohttp.ClientSession,
        snapshot: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Fetch bars for a single ticker and compute RSI; return candidate or None."""
        symbol: str = snapshot.get("ticker", "")
        if not symbol:
            return None

        async with self._semaphore:
            try:
                bars = await self._fetch_bars(session, symbol, self.bars_lookback + 5)
            except Exception as exc:
                logger.warning("Skipping %s — bar fetch failed: %s", symbol, exc)
                return None

        if len(bars) < self.rsi_period + 1:
            logger.debug("Skipping %s — insufficient bars (%d)", symbol, len(bars))
            return None

        closes = [float(b["c"]) for b in bars]
        rsi = _compute_rsi(closes, self.rsi_period)
        if rsi is None or rsi >= self.rsi_threshold:
            return None

        day = snapshot.get("day", {})
        return {
            "symbol": symbol,
            "price": float(snapshot.get("lastTrade", {}).get("p", closes[-1])),
            "volume": int(day.get("v", 0)),
            "rsi": round(rsi, 2),
            "market_cap": float(snapshot.get("market_cap", 0.0)),
        }
