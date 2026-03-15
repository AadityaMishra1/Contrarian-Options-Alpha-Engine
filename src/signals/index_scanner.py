"""Index Options Signal Scanner.

Computes the HV-IV gap for index ETFs (SPY, QQQ, IWM) using VIX as
the direct measure of SPX implied volatility. For SPY, IV = VIX/100.
For QQQ and IWM, fetches ATM IV from Polygon options snapshots.

The core signal: when VIX is elevated relative to SPX realized vol,
sell premium on index options (iron condors or strangles).

References:
- Coval & Shumway (2001): ATM straddle buyers lose ~3%/week on SPX
- Carr & Wu (2009): SPX variance risk premium is large and persistent
- Bondarenko (2014): ATM put buyers lose ~39%/month on SPX
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

# Index ETFs and their volatility index mappings
INDEX_UNIVERSE = {
    "SPY": {"vol_index": "VIX", "name": "S&P 500"},
    "QQQ": {"vol_index": None, "name": "Nasdaq 100"},  # use options snapshot
    "IWM": {"vol_index": None, "name": "Russell 2000"},  # use options snapshot
}


@dataclass
class IndexSignal:
    """Signal for an index ETF.

    Attributes:
        symbol: Index ETF ticker (SPY, QQQ, or IWM).
        price: Most recent closing price.
        hv_30: 30-day close-to-close realized vol, annualized.
        iv_30: 30-day implied vol (VIX/100 for SPY; ATM snapshot for others).
        gap_ratio: iv_30 / hv_30 — the Goyal-Saretto premium-selling signal.
        vix: Current VIX level (percentage points).
        regime: Regime label from RegimeClassifier.
        position_scalar: Regime-adjusted position size multiplier.
        skew_scalar: Skew-adjusted position size multiplier.
        effective_scalar: Combined regime * skew multiplier.
        structure: Recommended structure: "iron_condor", "strangle", or "put_spread".
        target_dte: Recommended DTE based on current regime.
        score: Composite attractiveness score (higher = more premium to sell).
    """

    symbol: str
    price: float
    hv_30: float
    iv_30: float
    gap_ratio: float
    vix: float
    regime: str
    position_scalar: float
    skew_scalar: float
    effective_scalar: float
    structure: str
    target_dte: int
    score: float


class IndexScanner:
    """Scans index ETFs for premium-selling opportunities.

    Uses VIX as the direct IV measure for SPY (VIX literally is SPX 30-day IV).
    For QQQ and IWM, fetches ATM IV from Polygon options snapshots.

    The scanner produces IndexSignal objects sorted by composite score. Callers
    should filter on ``signal.score > 0`` and ``signal.effective_scalar > 0.1``
    before acting.

    Args:
        api_key: Polygon.io API key for price data and options snapshots.
            Falls back to ``POLYGON_API_KEY`` environment variable.
        vix_csv_path: Path to CBOE VIX history CSV with a ``CLOSE`` column.
            Used as the primary VIX source when Polygon index data is unavailable
            on the Starter tier.
    """

    def __init__(
        self,
        api_key: str | None = None,
        vix_csv_path: str = "data/cache/vix_history.csv",
    ) -> None:
        resolved_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Polygon API key required. Set POLYGON_API_KEY env var "
                "or pass api_key explicitly."
            )
        self._api_key = resolved_key
        self._vix_csv_path = vix_csv_path
        self._vix_current: float = 0.0

    async def scan(
        self,
        regime_state: Any | None = None,
        skew_tracker: Any | None = None,
    ) -> list[IndexSignal]:
        """Scan index ETFs and return signals sorted by score descending.

        Args:
            regime_state: Current RegimeState from RegimeClassifier, used to
                set position_scalar and filter CRISIS regime.
            skew_tracker: SkewTracker instance for skew-based position scaling.

        Returns:
            List of IndexSignal for each tradeable index, best score first.
            Returns empty list if CRISIS regime is active.
        """
        signals: list[IndexSignal] = []

        async with aiohttp.ClientSession() as session:
            # Sequential to respect Polygon Starter rate limit (5 req/min)
            for symbol, meta in INDEX_UNIVERSE.items():
                try:
                    result = await self._analyze_index(
                        session, symbol, meta, regime_state, skew_tracker
                    )
                except Exception as exc:
                    logger.error("Index analysis raised exception: %s", exc)
                    result = None
                if result is not None:
                    signals.append(result)
                await asyncio.sleep(13)  # rate limit: 5 req/min

        signals.sort(key=lambda s: s.score, reverse=True)
        return signals

    async def _analyze_index(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        meta: dict[str, Any],
        regime_state: Any | None,
        skew_tracker: Any | None,
    ) -> IndexSignal | None:
        """Analyze a single index ETF and produce a signal.

        Args:
            session: Shared aiohttp session.
            symbol: ETF ticker.
            meta: Metadata from INDEX_UNIVERSE (vol_index mapping, name).
            regime_state: Current RegimeState or None.
            skew_tracker: SkewTracker or None.

        Returns:
            IndexSignal if the symbol qualifies, else None.
        """
        try:
            # 1. Fetch daily bars for realized vol computation
            bars = await self._get_bars(session, symbol, days=60)
            if len(bars) < 35:
                logger.warning(
                    "%s: insufficient bars (%d). Need at least 35.", symbol, len(bars)
                )
                return None

            closes = [b["c"] for b in bars]
            price = closes[-1]
            hv_30 = self._realized_vol(closes, 30)

            if hv_30 <= 0.01:
                logger.warning("%s: HV too low (%.4f). Skipping.", symbol, hv_30)
                return None

            # 2. Get implied volatility
            if symbol == "SPY":
                # VIX IS SPX 30-day IV. Direct. No proxy.
                # VIX is quoted in percentage points; convert to decimal.
                iv_30_raw = await self._get_vix_live(session)
                if iv_30_raw is None or iv_30_raw <= 0:
                    logger.warning("SPY: VIX unavailable. Skipping.")
                    return None
                iv_30 = iv_30_raw / 100.0
            else:
                # QQQ / IWM: fetch ATM IV from Polygon options snapshot
                iv_30 = await self._get_atm_iv(session, symbol)
                if iv_30 is None or iv_30 <= 0:
                    logger.warning(
                        "%s: ATM IV unavailable from options snapshot. Skipping.",
                        symbol,
                    )
                    return None

            gap_ratio = iv_30 / hv_30

            # 3. Regime classification
            regime_name = "NORMAL"
            position_scalar = 1.0
            if regime_state is not None:
                regime_name = regime_state.regime.value
                position_scalar = regime_state.position_scalar

            # Do not generate signals in CRISIS regime
            if regime_name == "CRISIS":
                return None

            # 4. Skew scalar
            skew_scalar = 1.0
            if skew_tracker is not None:
                skew_scalar = skew_tracker.get_position_scalar(symbol)

            effective_scalar = position_scalar * skew_scalar

            # 5. Recommend structure and DTE based on regime
            structure, target_dte = self._recommend_structure(regime_name)

            # 6. Compute composite attractiveness score
            score = self._compute_score(gap_ratio, effective_scalar, regime_name)

            return IndexSignal(
                symbol=symbol,
                price=price,
                hv_30=hv_30,
                iv_30=iv_30,
                gap_ratio=gap_ratio,
                vix=self._vix_current,
                regime=regime_name,
                position_scalar=position_scalar,
                skew_scalar=skew_scalar,
                effective_scalar=effective_scalar,
                structure=structure,
                target_dte=target_dte,
                score=score,
            )

        except Exception as exc:
            logger.error("Failed to analyze %s: %s", symbol, exc, exc_info=True)
            return None

    async def _get_vix_live(self, session: aiohttp.ClientSession) -> float | None:
        """Read current VIX from the CBOE history CSV.

        Polygon Starter tier does not provide index data (VIX ticker), so
        we fall back to a locally cached CBOE CSV. The CSV must have a
        ``CLOSE`` column with daily VIX closes (newest row last).

        Args:
            session: Unused; kept for signature consistency.

        Returns:
            Latest VIX close as a float (percentage points, e.g. 18.5),
            or None if the CSV cannot be read.
        """
        try:
            import pandas as pd

            df = pd.read_csv(self._vix_csv_path)
            if df.empty:
                return None
            close_col = "CLOSE" if "CLOSE" in df.columns else df.columns[-1]
            vix_value = float(df.iloc[-1][close_col])
            self._vix_current = vix_value
            return vix_value
        except Exception as exc:
            logger.warning("Could not read VIX CSV (%s): %s", self._vix_csv_path, exc)
            return None

    async def _get_bars(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        days: int = 60,
    ) -> list[dict[str, Any]]:
        """Fetch daily OHLCV bars from Polygon.

        Args:
            session: Active aiohttp session.
            symbol: Ticker symbol.
            days: Calendar days of history to request. A 1.5x buffer is
                applied to account for weekends and holidays.

        Returns:
            List of bar dicts with keys ``c`` (close), ``v`` (volume), etc.
            Returns empty list on any error.
        """
        end = date.today()
        start = end - timedelta(days=int(days * 1.5))
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}"
            f"/range/1/day/{start.isoformat()}/{end.isoformat()}"
        )
        params = {"apiKey": self._api_key, "limit": "200", "sort": "asc"}
        try:
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    logger.debug(
                        "Polygon bars returned %d for %s", resp.status, symbol
                    )
                    return []
                data = await resp.json()
            return data.get("results", [])
        except Exception as exc:
            logger.debug("Bar fetch failed for %s: %s", symbol, exc)
            return []

    async def _get_atm_iv(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
    ) -> float | None:
        """Fetch ATM implied volatility from the Polygon options snapshot.

        Selects options with 20-45 DTE whose absolute delta is between
        0.40 and 0.60 (near ATM) and averages their implied vols.

        Args:
            session: Active aiohttp session.
            symbol: Underlying ETF ticker (QQQ or IWM).

        Returns:
            Average ATM IV as a decimal (e.g. 0.22 for 22%), or None if
            no qualifying options are found or the request fails.
        """
        url = f"https://api.polygon.io/v3/snapshot/options/{symbol}"
        params = {"apiKey": self._api_key, "limit": "50"}
        try:
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    logger.debug(
                        "Polygon options snapshot returned %d for %s",
                        resp.status,
                        symbol,
                    )
                    return None
                data = await resp.json()

            today = date.today()
            atm_ivs: list[float] = []

            for opt in data.get("results", []):
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
                    atm_ivs.append(float(iv))

            if not atm_ivs:
                return None
            return sum(atm_ivs) / len(atm_ivs)

        except Exception as exc:
            logger.debug("ATM IV fetch failed for %s: %s", symbol, exc)
            return None

    @staticmethod
    def _realized_vol(closes: list[float], window: int) -> float:
        """Compute close-to-close realized volatility, annualized.

        Args:
            closes: List of daily closing prices in chronological order.
            window: Number of log-return periods to use.

        Returns:
            Annualized realized volatility as a decimal (e.g. 0.18 for 18%).
            Returns 0.0 if there are insufficient data points.
        """
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
        return math.sqrt(variance * 252)

    @staticmethod
    def _recommend_structure(regime: str) -> tuple[str, int]:
        """Recommend option structure and target DTE for the current regime.

        Args:
            regime: Regime label (QUIET, NORMAL, RECOVERY, ELEVATED, CRISIS).

        Returns:
            Tuple of (structure_name, target_dte).
            CRISIS returns ("none", 0) — caller should filter before using.
        """
        mapping: dict[str, tuple[str, int]] = {
            "QUIET": ("iron_condor", 30),      # thin premium; wide wings for safety
            "NORMAL": ("iron_condor", 30),     # standard carry trade
            "RECOVERY": ("strangle", 21),      # fat premium, mean-reverting VIX
            "ELEVATED": ("iron_condor", 14),   # defined risk required; shorter DTE
            "CRISIS": ("none", 0),             # no new entries
        }
        return mapping.get(regime, ("iron_condor", 30))

    @staticmethod
    def _compute_score(
        gap_ratio: float,
        effective_scalar: float,
        regime: str,
    ) -> float:
        """Compute composite attractiveness score for selling premium.

        Higher score = more attractive opportunity.

        Scoring logic:
        - base: how much IV exceeds HV, normalized so 0 = fair value.
          At gap_ratio=0.9 (IV slightly below HV), base=0.
          At gap_ratio=1.9 (IV 90% above HV), base=2.0.
        - regime_bonus: RECOVERY regime is the best risk/reward regime
          for selling vol (high VIX falling = fat premium + mean-reversion).
          ELEVATED gets a slight penalty (defined-risk constraint).
        - effective_scalar: Multiplier from regime + skew filters. A CRISIS
          regime yields scalar=0, naturally zeroing the score.

        Args:
            gap_ratio: iv_30 / hv_30.
            effective_scalar: Combined regime * skew position multiplier.
            regime: Regime label string.

        Returns:
            Non-negative float score. Zero means no opportunity.
        """
        # Base: IV premium above realized vol (floors at 0 if IV < 0.9 * HV)
        base = max(0.0, gap_ratio - 0.9) * 2.0

        # Regime-specific adjustment
        regime_bonus: dict[str, float] = {
            "RECOVERY": 0.5,   # best: high VIX falling
            "NORMAL": 0.2,     # standard
            "QUIET": 0.0,      # thin premium, no bonus
            "ELEVATED": -0.2,  # half-size, defined-risk constraint
            "CRISIS": -10.0,   # should never reach here (filtered upstream)
        }
        bonus = regime_bonus.get(regime, -1.0)

        return max(0.0, (base + bonus) * effective_scalar)
