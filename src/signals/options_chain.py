"""Options chain analyser using Polygon.io options endpoints.

Fetches live options chain data for a given underlying, applies the
strategy's DTE / delta / IV-percentile / spread filters, and scores
surviving candidates using a composite proximity-to-ideal formula.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_POLYGON_BASE = "https://api.polygon.io"
_OPTIONS_SNAPSHOT_ENDPOINT = "/v3/snapshot/options/{underlying}"
_OPTIONS_REF_ENDPOINT = "/v3/reference/options/contracts"

_DELTA_IDEAL = 0.30  # ideal absolute delta for the strategy


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class OptionCandidate:
    """A filtered and scored options contract candidate.

    Attributes:
        symbol: OCC option symbol (e.g. ``"O:AAPL251219C00150000"``).
        underlying: Underlying equity ticker.
        option_type: ``"call"`` or ``"put"``.
        strike: Strike price in dollars.
        expiry: Expiration date string (``YYYY-MM-DD``).
        dte: Days to expiration from today.
        bid: Bid price.
        ask: Ask price.
        delta: Absolute delta value.
        iv_percentile: Implied volatility percentile (0–100).
        mid_price: (bid + ask) / 2.
        spread_pct: (ask - bid) / mid * 100.
        score: Composite quality score in [0.0, 1.0]; higher is better.
    """

    symbol: str
    underlying: str
    option_type: str
    strike: float
    expiry: str
    dte: int
    bid: float
    ask: float
    delta: float
    iv_percentile: float
    mid_price: float
    spread_pct: float
    score: float = field(default=0.0)


# ---------------------------------------------------------------------------
# OptionsChainAnalyzer
# ---------------------------------------------------------------------------


class OptionsChainAnalyzer:
    """Fetches, filters, and scores options chain data from Polygon.io.

    Applies the contrarian strategy's criteria for DTE, delta, IV percentile,
    and bid-ask spread before computing a composite score for each contract.

    Args:
        api_key: Polygon.io API key. Falls back to ``POLYGON_API_KEY`` env var.
        dte_min: Minimum days-to-expiration (inclusive).
        dte_max: Maximum days-to-expiration (inclusive).
        delta_min: Minimum absolute delta (inclusive).
        delta_max: Maximum absolute delta (inclusive).
        iv_pct_max: Maximum allowed IV percentile (exclusive).
        spread_pct_max: Maximum allowed bid-ask spread as a percentage of mid.
    """

    def __init__(
        self,
        api_key: str | None = None,
        dte_min: int = 1,
        dte_max: int = 5,
        delta_min: float = 0.20,
        delta_max: float = 0.40,
        iv_pct_max: float = 50.0,
        spread_pct_max: float = 20.0,
    ) -> None:
        resolved_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Polygon API key is required. Pass api_key= or set the "
                "POLYGON_API_KEY environment variable."
            )
        self._api_key = resolved_key
        self.dte_min = dte_min
        self.dte_max = dte_max
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.iv_pct_max = iv_pct_max
        self.spread_pct_max = spread_pct_max
        self._semaphore = asyncio.Semaphore(5)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_chain(self, symbol: str) -> list[OptionCandidate]:
        """Fetch, filter, and score the options chain for *symbol*.

        Args:
            symbol: Underlying equity ticker (e.g. ``"AAPL"``).

        Returns:
            List of ``OptionCandidate`` objects sorted by score descending.
        """
        async with aiohttp.ClientSession() as session:
            raw_contracts = await self._fetch_snapshot(session, symbol)

        logger.info(
            "Fetched %d raw contracts for %s", len(raw_contracts), symbol
        )

        candidates: list[OptionCandidate] = []
        today = date.today()

        for contract in raw_contracts:
            candidate = self._parse_contract(contract, symbol, today)
            if candidate is None:
                continue
            if not self._passes_filters(candidate):
                continue
            candidate.score = self._compute_score(candidate)
            candidates.append(candidate)

        candidates.sort(key=lambda c: c.score, reverse=True)
        logger.info(
            "%d candidates passed filters for %s", len(candidates), symbol
        )
        return candidates

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_snapshot(
        self, session: aiohttp.ClientSession, symbol: str
    ) -> list[dict[str, Any]]:
        """Fetch the options snapshot from Polygon for *symbol*.

        Handles pagination via the ``next_url`` field.

        Args:
            session: Active aiohttp session.
            symbol: Underlying ticker.

        Returns:
            Flat list of contract snapshot dicts.
        """
        url = f"{_POLYGON_BASE}{_OPTIONS_SNAPSHOT_ENDPOINT.format(underlying=symbol)}"
        params: dict[str, str] = {
            "apiKey": self._api_key,
            "limit": "250",
        }

        results: list[dict[str, Any]] = []
        pages_fetched = 0
        max_pages = 10  # guard against runaway pagination

        async with self._semaphore:
            while url and pages_fetched < max_pages:
                try:
                    async with session.get(
                        url,
                        params=params if pages_fetched == 0 else None,
                        timeout=aiohttp.ClientTimeout(total=20),
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()

                    results.extend(data.get("results", []))
                    pages_fetched += 1

                    next_url: str | None = data.get("next_url")
                    if next_url:
                        # Polygon appends the cursor; we only need to add apiKey
                        url = next_url + f"&apiKey={self._api_key}"
                    else:
                        break

                except Exception as exc:
                    logger.error(
                        "Failed to fetch options snapshot for %s (page %d): %s",
                        symbol, pages_fetched, exc,
                    )
                    break

        return results

    def _parse_contract(
        self,
        raw: dict[str, Any],
        underlying: str,
        today: date,
    ) -> OptionCandidate | None:
        """Parse a Polygon snapshot contract dict into an OptionCandidate.

        Args:
            raw: Raw contract dict from the Polygon snapshot response.
            underlying: Underlying ticker string.
            today: Today's date for DTE calculation.

        Returns:
            ``OptionCandidate`` or None if required fields are missing/invalid.
        """
        try:
            details: dict[str, Any] = raw.get("details", {})
            greeks: dict[str, Any] = raw.get("greeks", {})
            day: dict[str, Any] = raw.get("day", {})
            last_quote: dict[str, Any] = raw.get("last_quote", {})

            option_symbol: str = details.get("ticker", raw.get("ticker", ""))
            if not option_symbol:
                return None

            option_type: str = details.get("contract_type", "").lower()
            if option_type not in ("call", "put"):
                return None

            strike: float = float(details.get("strike_price", 0.0))
            expiry_str: str = details.get("expiration_date", "")
            if not expiry_str:
                return None

            expiry_date = date.fromisoformat(expiry_str)
            dte: int = (expiry_date - today).days
            if dte < 0:
                return None

            bid: float = float(last_quote.get("bid", day.get("open", 0.0)))
            ask: float = float(last_quote.get("ask", day.get("close", 0.0)))

            if bid <= 0.0 or ask <= 0.0 or ask < bid:
                return None

            mid_price: float = (bid + ask) / 2.0
            spread_pct: float = ((ask - bid) / mid_price * 100.0) if mid_price > 0 else 100.0

            delta: float = abs(float(greeks.get("delta", 0.0)))
            iv_percentile: float = float(
                raw.get("implied_volatility_percentile",
                        raw.get("iv_percentile", 100.0))
            )

            return OptionCandidate(
                symbol=option_symbol,
                underlying=underlying,
                option_type=option_type,
                strike=strike,
                expiry=expiry_str,
                dte=dte,
                bid=bid,
                ask=ask,
                delta=delta,
                iv_percentile=iv_percentile,
                mid_price=round(mid_price, 4),
                spread_pct=round(spread_pct, 2),
            )

        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("Skipping malformed contract: %s — %s", raw.get("ticker"), exc)
            return None

    def _passes_filters(self, c: OptionCandidate) -> bool:
        """Return True when *c* satisfies all strategy filter criteria."""
        if not (self.dte_min <= c.dte <= self.dte_max):
            return False
        if not (self.delta_min <= c.delta <= self.delta_max):
            return False
        if c.iv_percentile >= self.iv_pct_max:
            return False
        return not c.spread_pct > self.spread_pct_max

    def _compute_score(self, c: OptionCandidate) -> float:
        """Compute a composite quality score in [0.0, 1.0] for *c*.

        Score components:
        - Delta proximity: how close |delta| is to the ideal 0.30.
        - IV discount: lower IV percentile is better.
        - Spread tightness: lower spread percentage is better.

        All three components are equally weighted (1/3 each).

        Args:
            c: Filtered ``OptionCandidate``.

        Returns:
            Composite score in [0.0, 1.0].
        """
        delta_range = (self.delta_max - self.delta_min) / 2.0
        delta_ideal = _DELTA_IDEAL
        delta_score = max(
            0.0, 1.0 - abs(c.delta - delta_ideal) / max(delta_range, 1e-9)
        )

        iv_score = max(0.0, 1.0 - c.iv_percentile / max(self.iv_pct_max, 1e-9))

        spread_score = max(
            0.0, 1.0 - c.spread_pct / max(self.spread_pct_max, 1e-9)
        )

        return round((delta_score + iv_score + spread_score) / 3.0, 4)
