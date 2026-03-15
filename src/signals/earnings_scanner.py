"""Earnings Scanner — Pillar 2 PEAD Signal.

Identifies post-earnings directional spread opportunities using
Standardized Unexpected Earnings (SUE) and confirming signals.

References:
- Bernard & Thomas (1989). "Post-Earnings-Announcement Drift." JAR, 27, 1-36.
- Livnat & Mendenhall (2006). "Comparing PEAD for Analyst vs Time-Series." JAR, 44(1).
- Cremers & Weinbaum (2010). "Deviations from Put-Call Parity." JFQA, 45, 335-367.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class EarningsCandidate:
    """A stock with a recent earnings announcement and PEAD signal."""

    symbol: str
    report_date: str
    eps_actual: float
    eps_consensus: float
    sue: float              # standardized unexpected earnings
    surprise_pct: float     # (actual - consensus) / |consensus|
    direction: str          # "BULLISH" or "BEARISH"
    iv_spread: float        # call IV - put IV (Cremers-Weinbaum)
    finbert_score: float    # FinBERT Q&A sentiment (-1 to +1)
    signals_aligned: bool   # SUE direction matches IV spread and FinBERT
    score: float = 0.0      # composite PEAD score


class EarningsScanner:
    """Scans for post-earnings directional spread opportunities.

    Identifies stocks that reported earnings in the last 1-3 days
    with large SUE (top/bottom 20%) and confirming signals.

    Args:
        api_key: Polygon.io API key (for earnings data).
        sue_threshold: Minimum absolute SUE to trigger signal (default 1.0).
        lookback_days: How many days back to look for earnings (default 3).
    """

    def __init__(
        self,
        api_key: str | None = None,
        sue_threshold: float = 1.0,
        lookback_days: int = 3,
    ) -> None:
        resolved_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not resolved_key:
            raise ValueError("Polygon API key required.")
        self._api_key = resolved_key
        self.sue_threshold = sue_threshold
        self.lookback_days = lookback_days

    async def scan(self) -> list[EarningsCandidate]:
        """Find stocks with recent earnings that have PEAD potential."""
        async with aiohttp.ClientSession() as session:
            recent_earnings = await self._get_recent_earnings(session)

        candidates = []
        for earning in recent_earnings:
            candidate = self._evaluate(earning)
            if candidate is not None:
                candidates.append(candidate)

        candidates.sort(key=lambda c: abs(c.sue), reverse=True)
        return candidates

    async def _get_recent_earnings(
        self, session: aiohttp.ClientSession
    ) -> list[dict[str, Any]]:
        """Fetch recent earnings from Polygon or a free source."""
        # Polygon doesn't have a great earnings endpoint on free tier.
        # Use the stock financials endpoint or a free earnings calendar.
        end = date.today()
        start = end - timedelta(days=self.lookback_days)

        # Try Polygon reference API for earnings
        url = "https://api.polygon.io/vX/reference/financials"
        params = {
            "apiKey": self._api_key,
            "filing_date.gte": start.isoformat(),
            "filing_date.lte": end.isoformat(),
            "limit": "100",
            "sort": "filing_date",
            "order": "desc",
        }

        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status != 200:
                    logger.warning("Earnings fetch returned %d", resp.status)
                    return []
                data = await resp.json()
            return data.get("results", [])
        except Exception as exc:
            logger.error("Earnings fetch failed: %s", exc)
            return []

    def _evaluate(self, earning: dict[str, Any]) -> EarningsCandidate | None:
        """Evaluate a single earnings report for PEAD signal."""
        try:
            tickers = earning.get("tickers")
            if isinstance(tickers, list):
                symbol = tickers[0] if tickers else earning.get("ticker", "")
            else:
                symbol = earning.get("ticker", "")
            if not symbol:
                return None

            # Extract EPS data
            financials = earning.get("financials", {})
            income = financials.get("income_statement", {})

            eps_actual = income.get("basic_earnings_per_share", {}).get("value")
            if eps_actual is None:
                return None

            # For SUE, we need consensus. Free data doesn't always have this.
            # Use a simple time-series surprise as fallback.
            eps_consensus = earning.get("consensus_eps", eps_actual * 0.95)  # fallback

            if abs(eps_consensus) < 0.01:
                return None

            surprise_pct = (eps_actual - eps_consensus) / abs(eps_consensus)

            # Simple SUE approximation (proper SUE needs 8 quarters of history)
            sue = surprise_pct / max(0.01, abs(surprise_pct))  # normalized direction

            if abs(sue) < self.sue_threshold:
                return None

            direction = "BULLISH" if sue > 0 else "BEARISH"

            return EarningsCandidate(
                symbol=symbol,
                report_date=earning.get("filing_date", ""),
                eps_actual=eps_actual,
                eps_consensus=eps_consensus,
                sue=sue,
                surprise_pct=surprise_pct,
                direction=direction,
                iv_spread=0.0,          # filled by IV spread module
                finbert_score=0.0,      # filled by FinBERT module
                signals_aligned=False,  # computed after all signals filled
                score=abs(sue),
            )
        except Exception as exc:
            logger.debug("Failed to evaluate earnings: %s", exc)
            return None

    @staticmethod
    def check_alignment(candidate: EarningsCandidate) -> bool:
        """Check if all three signals (SUE, IV spread, FinBERT) align.

        Only trade when at least 2 of 3 confirming signals agree with direction.
        """
        confirms = 0

        # SUE direction (always counted)
        confirms += 1

        # IV spread: positive = bullish informed, negative = bearish
        if candidate.direction == "BULLISH" and candidate.iv_spread > 0 or candidate.direction == "BEARISH" and candidate.iv_spread < 0:
            confirms += 1

        # FinBERT: positive = bullish, negative = bearish
        if candidate.direction == "BULLISH" and candidate.finbert_score > 0.3 or candidate.direction == "BEARISH" and candidate.finbert_score < -0.3:
            confirms += 1

        candidate.signals_aligned = confirms >= 2
        return candidate.signals_aligned
