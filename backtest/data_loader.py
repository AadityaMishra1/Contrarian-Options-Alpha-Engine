"""Polygon.io data loader with Parquet caching for options backtesting.

Provides bar data and options chain snapshots for a given underlying symbol.
Falls back to synthetic mean-reverting price data when no API key is present.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"


def _cache_path(symbol: str, start: str, end: str, suffix: str = "") -> Path:
    """Build a deterministic Parquet cache file path."""
    tag = f"{symbol}_{start}_{end}{suffix}".replace("-", "").replace("/", "_")
    return CACHE_DIR / f"{tag}.parquet"


def _generate_synthetic_bars(
    symbol: str,
    start: str,
    end: str,
    timespan: str = "minute",
) -> pd.DataFrame:
    """Generate a synthetic OHLCV bar series with mean-reversion.

    Uses an Ornstein-Uhlenbeck process centred on $150 so the backtest
    engine always has realistic-looking data even without a live API key.

    Args:
        symbol: Ticker symbol (used only to seed the RNG for reproducibility).
        start: ISO date string for the first bar.
        end: ISO date string for the last bar.
        timespan: Bar frequency; only ``"minute"`` and ``"day"`` are mapped.

    Returns:
        DataFrame with columns [timestamp, open, high, low, close, volume, vwap].
    """
    freq_map = {"minute": "1min", "hour": "1h", "day": "1D"}
    freq = freq_map.get(timespan, "1min")

    seed = sum(ord(c) for c in symbol)
    rng = np.random.default_rng(seed)

    idx = pd.date_range(start=start, end=end, freq=freq)
    n = len(idx)
    if n == 0:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "vwap"])

    # Ornstein-Uhlenbeck mean reversion
    theta, mu, sigma = 0.05, 150.0, 1.5
    prices = np.empty(n)
    prices[0] = mu
    for i in range(1, n):
        prices[i] = prices[i - 1] + theta * (mu - prices[i - 1]) + sigma * rng.standard_normal()

    prices = np.maximum(prices, 1.0)  # prevent non-positive prices
    noise = rng.uniform(0.0, 0.5, (n, 2))
    highs = prices + noise[:, 0]
    lows = prices - noise[:, 1]
    opens = (prices + np.roll(prices, 1)) / 2
    opens[0] = prices[0]
    volumes = (rng.integers(100_000, 5_000_000, n)).astype(float)

    # Inject volume spikes ~2% of the time to trigger strategy signals
    spike_idx = rng.choice(n, size=max(1, n // 50), replace=False)
    volumes[spike_idx] *= rng.uniform(3.0, 8.0, len(spike_idx))

    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volumes,
            "vwap": (highs + lows + prices) / 3,
        }
    )
    return df


def _generate_synthetic_options_chain(
    underlying: str,
    expiry_date: str,
    option_type: str,
    spot: float = 150.0,
) -> pd.DataFrame:
    """Generate a synthetic options chain for a given expiry.

    Produces a grid of strikes around the spot price with synthetic bid/ask
    prices consistent with Black-Scholes intuition (deeper OTM = cheaper).

    Args:
        underlying: Underlying ticker symbol.
        expiry_date: Target expiry date as an ISO string.
        option_type: ``"call"`` or ``"put"``.
        spot: Current underlying price used to centre the strike grid.

    Returns:
        DataFrame with columns [strike, bid, ask, mid, volume, open_interest,
        implied_volatility, expiry, option_type].
    """
    seed = sum(ord(c) for c in underlying + expiry_date)
    rng = np.random.default_rng(seed)

    strikes = np.arange(
        round(spot * 0.85 / 5) * 5,
        round(spot * 1.15 / 5) * 5 + 5,
        5,
    ).astype(float)

    rows = []
    for strike in strikes:
        moneyness = (spot - strike) / spot if option_type == "put" else (strike - spot) / spot
        # Very rough proxy: deeper OTM -> lower price
        base_price = max(0.01, 0.5 * np.exp(-3 * max(moneyness, 0.0)))
        iv = rng.uniform(0.20, 0.60)
        spread = max(0.01, base_price * 0.15)
        bid = max(0.01, base_price - spread / 2)
        ask = bid + spread
        mid = (bid + ask) / 2
        rows.append(
            {
                "strike": strike,
                "bid": round(bid, 2),
                "ask": round(ask, 2),
                "mid": round(mid, 2),
                "volume": int(rng.integers(0, 500)),
                "open_interest": int(rng.integers(100, 5000)),
                "implied_volatility": round(iv, 4),
                "expiry": expiry_date,
                "option_type": option_type,
            }
        )
    return pd.DataFrame(rows)


class PolygonDataLoader:
    """Load stock bars and options chains from Polygon.io with local caching.

    If ``POLYGON_API_KEY`` is not set in the environment, all methods return
    synthetic data suitable for strategy development and testing.

    Args:
        api_key: Polygon.io API key. Defaults to the ``POLYGON_API_KEY``
                 environment variable.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key: str | None = api_key or os.environ.get("POLYGON_API_KEY")
        self._client = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        if self.api_key:
            try:
                from polygon import RESTClient  # type: ignore[import]

                self._client = RESTClient(api_key=self.api_key)
                logger.info("Polygon REST client initialised.")
            except ImportError:
                logger.warning("polygon-api-client not installed; using synthetic data.")
        else:
            logger.info("No POLYGON_API_KEY found — synthetic data mode active.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_stock_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        timespan: str = "minute",
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for a stock, using Parquet cache when available.

        Args:
            symbol: Equity ticker (e.g. ``"AAPL"``).
            start: Start date as ``"YYYY-MM-DD"``.
            end: End date as ``"YYYY-MM-DD"``.
            timespan: Bar granularity: ``"minute"``, ``"hour"``, or ``"day"``.

        Returns:
            DataFrame with columns [timestamp, open, high, low, close, volume, vwap].
            Sorted ascending by timestamp.
        """
        cache = _cache_path(symbol, start, end, f"_{timespan}")
        if cache.exists():
            logger.debug("Cache hit: %s", cache)
            return pd.read_parquet(cache)

        if self._client is None:
            logger.info("Generating synthetic bars for %s [%s – %s].", symbol, start, end)
            df = _generate_synthetic_bars(symbol, start, end, timespan)
        else:
            df = self._fetch_polygon_bars(symbol, start, end, timespan)

        if not df.empty:
            df.to_parquet(cache, index=False)
        return df

    def load_options_chain(
        self,
        underlying: str,
        expiry_date: str,
        option_type: str,
    ) -> pd.DataFrame:
        """Fetch the options chain for a given underlying and expiry.

        Args:
            underlying: Equity ticker of the underlying (e.g. ``"AAPL"``).
            expiry_date: Expiry date as ``"YYYY-MM-DD"``.
            option_type: ``"call"`` or ``"put"``.

        Returns:
            DataFrame with columns [strike, bid, ask, mid, volume,
            open_interest, implied_volatility, expiry, option_type].
        """
        cache = _cache_path(underlying, expiry_date, expiry_date, f"_chain_{option_type}")
        if cache.exists():
            return pd.read_parquet(cache)

        if self._client is None:
            df = _generate_synthetic_options_chain(underlying, expiry_date, option_type)
        else:
            df = self._fetch_polygon_options(underlying, expiry_date, option_type)

        if not df.empty:
            df.to_parquet(cache, index=False)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_polygon_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        timespan: str,
    ) -> pd.DataFrame:
        """Retrieve aggregate bars from Polygon REST API with retry logic.

        Args:
            symbol: Equity ticker.
            start: Start date string.
            end: End date string.
            timespan: Bar granularity.

        Returns:
            Normalised OHLCV DataFrame.
        """
        max_retries = 5
        for attempt in range(max_retries):
            try:
                aggs = list(
                    self._client.list_aggs(  # type: ignore[union-attr]
                        ticker=symbol,
                        multiplier=1,
                        timespan=timespan,
                        from_=start,
                        to=end,
                        limit=50000,
                    )
                )
                if not aggs:
                    logger.warning("Polygon returned no bars for %s.", symbol)
                    return pd.DataFrame()

                rows = [
                    {
                        "timestamp": pd.Timestamp(a.timestamp, unit="ms", tz="UTC"),
                        "open": a.open,
                        "high": a.high,
                        "low": a.low,
                        "close": a.close,
                        "volume": a.volume,
                        "vwap": getattr(a, "vwap", (a.high + a.low + a.close) / 3),
                    }
                    for a in aggs
                ]
                return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

            except Exception as exc:  # noqa: BLE001
                wait = 2**attempt
                logger.warning(
                    "Polygon request failed (attempt %d/%d): %s. Retrying in %ds.",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait,
                )
                if attempt < max_retries - 1:
                    time.sleep(wait)
                else:
                    logger.error("All retries exhausted for %s.", symbol)
                    return pd.DataFrame()

        return pd.DataFrame()

    def _fetch_polygon_options(
        self,
        underlying: str,
        expiry_date: str,
        option_type: str,
    ) -> pd.DataFrame:
        """Retrieve options chain snapshot from Polygon.

        Args:
            underlying: Equity ticker.
            expiry_date: Target expiry.
            option_type: ``"call"`` or ``"put"``.

        Returns:
            Options chain DataFrame.
        """
        max_retries = 5
        contract_type = "call" if option_type.lower() == "call" else "put"
        for attempt in range(max_retries):
            try:
                contracts = list(
                    self._client.list_snapshot_options_chain(  # type: ignore[union-attr]
                        underlying,
                        params={
                            "expiration_date": expiry_date,
                            "contract_type": contract_type,
                            "limit": 250,
                        },
                    )
                )
                if not contracts:
                    return _generate_synthetic_options_chain(underlying, expiry_date, option_type)

                rows = []
                for c in contracts:
                    details = getattr(c, "details", None)
                    greeks = getattr(c, "greeks", None)
                    day = getattr(c, "day", None)
                    if details is None:
                        continue
                    bid = getattr(c, "last_quote", None)
                    bid_price = getattr(bid, "bid", 0.0) if bid else 0.0
                    ask_price = getattr(bid, "ask", 0.0) if bid else 0.0
                    mid = (bid_price + ask_price) / 2
                    rows.append(
                        {
                            "strike": details.strike_price,
                            "bid": bid_price,
                            "ask": ask_price,
                            "mid": mid,
                            "volume": getattr(day, "volume", 0) if day else 0,
                            "open_interest": getattr(c, "open_interest", 0),
                            "implied_volatility": getattr(greeks, "iv", 0.30) if greeks else 0.30,
                            "expiry": expiry_date,
                            "option_type": option_type,
                        }
                    )
                return pd.DataFrame(rows) if rows else _generate_synthetic_options_chain(
                    underlying, expiry_date, option_type
                )

            except Exception as exc:  # noqa: BLE001
                wait = 2**attempt
                logger.warning("Options chain request failed (attempt %d): %s", attempt + 1, exc)
                if attempt < max_retries - 1:
                    time.sleep(wait)
                else:
                    return _generate_synthetic_options_chain(underlying, expiry_date, option_type)

        return pd.DataFrame()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Test PolygonDataLoader")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-01-31")
    parser.add_argument("--timespan", default="day")
    args = parser.parse_args()

    loader = PolygonDataLoader()
    df = loader.load_stock_bars(args.symbol, args.start, args.end, args.timespan)
    print(f"Loaded {len(df)} bars for {args.symbol}")
    print(df.tail())
