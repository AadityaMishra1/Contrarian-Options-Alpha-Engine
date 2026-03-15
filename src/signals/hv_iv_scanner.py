"""HV-IV Gap Scanner — Primary signal for the v2 strategy.

Implements the Goyal-Saretto (2009) cross-sectional signal: sort stocks by
the ratio of implied volatility to historical realized volatility, and
identify candidates where options are overpriced (IV >> HV) for selling.

Reference: Goyal, A. and Saretto, A. (2009). "Cross-Section of Option Returns
and Volatility." Journal of Financial Economics, 94(2), 310-326.
"""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class HVIVCandidate:
    """A stock with computed HV-IV gap metrics."""

    symbol: str
    price: float
    hv_30: float          # 30-day close-to-close realized vol (annualized)
    hv_60: float          # 60-day realized vol (for robustness)
    iv_30: float          # 30-day ATM implied vol
    gap_ratio: float      # iv_30 / hv_30
    gap_zscore: float     # z-score of gap_ratio vs 60-day rolling history
    quintile: int         # 1-5, where 5 = most overpriced IV
    skew_zscore: float    # IV skew z-score (from skew module)
    volume: int           # daily options volume
    market_cap: float     # for filtering
    days_to_earnings: int # avoid near-earnings names for Pillar 1
    score: float = 0.0    # continuous attractiveness score


class HVIVScanner:
    """Scans a universe of stocks for HV-IV gap opportunities.

    Args:
        api_key: Polygon.io API key.
        universe_size: Number of top liquid stocks to scan.
        min_gap_ratio: Minimum IV/HV ratio to consider selling (default 1.2).
        min_volume: Minimum daily options volume.
        earnings_buffer: Skip stocks with earnings within N days.
    """

    def __init__(
        self,
        api_key: str | None = None,
        universe_size: int = 100,
        min_gap_ratio: float = 1.2,
        min_volume: int = 1000,
        earnings_buffer: int = 7,
    ) -> None:
        import os
        resolved_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not resolved_key:
            raise ValueError("Polygon API key required. Set POLYGON_API_KEY env var.")
        self._api_key = resolved_key
        self.universe_size = universe_size
        self.min_gap_ratio = min_gap_ratio
        self.min_volume = min_volume
        self.earnings_buffer = earnings_buffer
        self._semaphore = asyncio.Semaphore(5)

    async def scan(self) -> list[HVIVCandidate]:
        """Scan universe and return candidates sorted by gap_ratio descending."""
        async with aiohttp.ClientSession() as session:
            # 1. Get universe of liquid optionable stocks
            universe = await self._get_universe(session)

            # 2. For each, compute HV and fetch IV
            tasks = [self._analyze_stock(session, sym) for sym in universe]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates = []
        for r in results:
            if isinstance(r, Exception):
                continue
            if r is not None and r.gap_ratio >= self.min_gap_ratio:
                candidates.append(r)

        # Assign quintiles
        candidates.sort(key=lambda c: c.gap_ratio)
        n = len(candidates)
        for i, c in enumerate(candidates):
            c.quintile = min(5, (i * 5) // n + 1) if n > 0 else 3
            c.score = self._compute_score(c)

        # Return sorted by score descending (best sell candidates first)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    async def _get_universe(self, session: aiohttp.ClientSession) -> list[str]:
        """Get top liquid optionable stocks from Polygon snapshots."""
        url = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers"
        params = {"apiKey": self._api_key}
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    logger.error("Failed to fetch universe: %d", resp.status)
                    return []
                data = await resp.json()

            tickers = data.get("tickers", [])
            # Sort by volume, take top N
            tickers.sort(key=lambda t: t.get("day", {}).get("v", 0), reverse=True)
            return [
                t["ticker"] for t in tickers[:self.universe_size]
                if t.get("day", {}).get("v", 0) > 100000
            ]
        except Exception as exc:
            logger.error("Universe fetch failed: %s", exc)
            return []

    async def _analyze_stock(
        self, session: aiohttp.ClientSession, symbol: str
    ) -> HVIVCandidate | None:
        """Compute HV-IV metrics for a single stock."""
        async with self._semaphore:
            try:
                # Fetch 90 days of daily bars for HV calculation
                bars = await self._get_bars(session, symbol, days=90)
                if len(bars) < 60:
                    return None

                closes = [b["c"] for b in bars]
                price = closes[-1]

                # Compute realized vol (close-to-close, annualized)
                hv_30 = self._realized_vol(closes, 30)
                hv_60 = self._realized_vol(closes, 60)

                if hv_30 <= 0 or hv_60 <= 0:
                    return None

                # Fetch ATM IV from options snapshot
                iv_30 = await self._get_atm_iv(session, symbol)
                if iv_30 is None or iv_30 <= 0:
                    return None

                gap_ratio = iv_30 / hv_30

                # Z-score: compare current gap to rolling history
                # Simple approximation: use hv_60 as reference
                gap_60 = iv_30 / hv_60 if hv_60 > 0 else gap_ratio
                gap_zscore = (gap_ratio - gap_60) / max(0.01, abs(gap_60 - 1.0))

                return HVIVCandidate(
                    symbol=symbol,
                    price=price,
                    hv_30=hv_30,
                    hv_60=hv_60,
                    iv_30=iv_30,
                    gap_ratio=gap_ratio,
                    gap_zscore=gap_zscore,
                    quintile=0,        # assigned later
                    skew_zscore=0.0,   # filled by skew module
                    volume=bars[-1].get("v", 0) if bars else 0,
                    market_cap=0.0,
                    days_to_earnings=999,  # filled by earnings module
                )
            except Exception as exc:
                logger.debug("Failed to analyze %s: %s", symbol, exc)
                return None

    async def _get_bars(
        self, session: aiohttp.ClientSession, symbol: str, days: int = 90
    ) -> list[dict[str, Any]]:
        """Fetch daily bars from Polygon."""
        end = date.today()
        start = end - timedelta(days=int(days * 1.5))  # buffer for weekends
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}"
            f"/range/1/day/{start.isoformat()}/{end.isoformat()}"
        )
        params = {"apiKey": self._api_key, "limit": "200", "sort": "asc"}
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
            return data.get("results", [])
        except Exception:
            return []

    async def _get_atm_iv(
        self, session: aiohttp.ClientSession, symbol: str
    ) -> float | None:
        """Fetch ATM implied volatility from Polygon options snapshot."""
        url = f"https://api.polygon.io/v3/snapshot/options/{symbol}"
        params = {"apiKey": self._api_key, "limit": "50"}
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()

            results = data.get("results", [])
            if not results:
                return None

            # Find options nearest to ATM with 20-45 DTE
            today = date.today()
            atm_ivs = []
            for opt in results:
                details = opt.get("details", {})
                greeks = opt.get("greeks", {})

                expiry_str = details.get("expiration_date", "")
                if not expiry_str:
                    continue
                dte = (date.fromisoformat(expiry_str) - today).days
                if not (20 <= dte <= 45):
                    continue

                delta = abs(greeks.get("delta", 0))
                iv = opt.get("implied_volatility", 0)
                if 0.40 <= delta <= 0.60 and iv > 0:
                    atm_ivs.append(iv)

            if not atm_ivs:
                return None
            return sum(atm_ivs) / len(atm_ivs)
        except Exception:
            return None

    @staticmethod
    def _realized_vol(closes: list[float], window: int) -> float:
        """Close-to-close realized volatility, annualized."""
        if len(closes) < window + 1:
            return 0.0
        recent = closes[-(window + 1):]
        log_returns = [
            math.log(recent[i + 1] / recent[i])
            for i in range(len(recent) - 1)
            if recent[i] > 0
        ]
        if len(log_returns) < 2:
            return 0.0
        mean_r = sum(log_returns) / len(log_returns)
        variance = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
        return math.sqrt(variance * 252)  # annualize

    def _compute_score(self, c: HVIVCandidate) -> float:
        """Continuous attractiveness score for selling options.

        Higher = more attractive to sell (IV more overpriced).
        Uses gap_ratio as primary, skew_zscore as penalty.
        """
        # Base: how overpriced is IV (capped at 3.0)
        base = min(3.0, c.gap_ratio - 1.0)  # 0 = fair, 1.0 = IV is 2x HV

        # Penalty for steep skew (informed bearish flow)
        skew_penalty = max(0.0, c.skew_zscore - 1.0) * 0.3

        # Penalty for near-earnings
        earnings_penalty = 0.5 if c.days_to_earnings < self.earnings_buffer else 0.0

        return max(0.0, base - skew_penalty - earnings_penalty)
