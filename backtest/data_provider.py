"""Historical Data Provider for backtesting.

Fetches and caches real historical price and options data from Polygon.io.
This is ESSENTIAL — the v2 strategy must be validated on real data, not
synthetic Ornstein-Uhlenbeck processes.

Survivorship bias note: when backtesting S&P 500, use historical constituent
lists (not current members) to avoid look-ahead bias. The SP500_LIQUID_50
constant below reflects CURRENT membership and is provided only as a
convenience; replace it with point-in-time constituent data for rigorous
research.

Usage:
    import asyncio
    from backtest.data_provider import PolygonDataProvider

    provider = PolygonDataProvider()          # reads POLYGON_API_KEY from env
    price_df = asyncio.run(
        provider.get_universe_bars(["AAPL", "MSFT"], "2020-01-01", "2025-01-01")
    )
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_POLYGON_BASE = "https://api.polygon.io"

# ---------------------------------------------------------------------------
# S&P 500 convenience universe (CURRENT constituents — not point-in-time)
# ---------------------------------------------------------------------------

#: Current liquid S&P 500 names used for rough universe construction.
#: WARNING: Using this list for long historical backtests introduces
#: survivorship bias because it excludes companies that were in the index
#: historically but have since been removed (e.g. due to bankruptcy, merger,
#: or demotion). For academic-quality backtests supply point-in-time lists
#: from a data vendor such as Compustat or CRSP.
SP500_LIQUID_50: list[str] = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK.B",
    "JPM", "JNJ", "V", "UNH", "HD", "PG", "MA", "DIS", "BAC", "XOM",
    "ADBE", "CRM", "NFLX", "COST", "PEP", "TMO", "ABT", "AVGO", "CSCO",
    "ACN", "MRK", "NKE", "WMT", "LLY", "ABBV", "DHR", "TXN", "ORCL",
    "PM", "QCOM", "UPS", "LOW", "NEE", "MS", "INTC", "AMD", "AMGN",
    "MDT", "HON", "SCHW", "GS", "CAT",
]


# ---------------------------------------------------------------------------
# Main provider
# ---------------------------------------------------------------------------


class PolygonDataProvider:
    """Fetches and caches historical price and IV data from Polygon.io.

    Results are persisted as Parquet files under ``cache_dir`` so that
    repeated backtest runs do not re-fetch the same date ranges.

    Args:
        api_key: Polygon.io API key. Falls back to the ``POLYGON_API_KEY``
                 environment variable when not supplied.
        cache_dir: Local directory for Parquet cache files. Created
                   automatically if it does not exist.

    Raises:
        ValueError: If no API key can be resolved.
        ImportError: If ``aiohttp`` is not installed (add it via
                     ``pip install aiohttp`` or the project's
                     ``backtest`` optional dependency group).

    Example:
        >>> provider = PolygonDataProvider()
        >>> df = asyncio.run(provider.get_daily_bars("AAPL", "2020-01-01", "2025-01-01"))
        >>> print(df.columns.tolist())
        ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str = "data/cache",
    ) -> None:
        resolved_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Polygon API key is required. "
                "Pass api_key= or set the POLYGON_API_KEY environment variable."
            )
        self._api_key = resolved_key
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Validate aiohttp is available at construction time so the error is
        # raised early rather than at the first async call.
        try:
            import aiohttp as _aiohttp  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "aiohttp is required for PolygonDataProvider. "
                "Install it with: pip install aiohttp"
            ) from exc

        # Rate limiter: Polygon free tier allows ~5 req/s; paid tiers are
        # higher, but 5 is a safe default that avoids 429 errors.
        self._semaphore = asyncio.Semaphore(5)

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def get_daily_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars for a single symbol.

        Args:
            symbol: Equity ticker (e.g. ``"AAPL"``).
            start: Start date in ``YYYY-MM-DD`` format (inclusive).
            end: End date in ``YYYY-MM-DD`` format (inclusive).
            use_cache: When True, return cached Parquet data if available
                       and skip the network call.

        Returns:
            DataFrame with columns
            ``[date, symbol, open, high, low, close, volume]``,
            sorted ascending by date. Returns an empty DataFrame when
            Polygon returns no results or the request fails.
        """
        import aiohttp

        cache_file = self._cache_dir / f"bars_{symbol}_{start}_{end}.parquet"

        if use_cache and cache_file.exists():
            logger.debug("Cache hit — loading bars for %s from %s", symbol, cache_file)
            return pd.read_parquet(cache_file)

        url = (
            f"{_POLYGON_BASE}/v2/aggs/ticker/{symbol}"
            f"/range/1/day/{start}/{end}"
        )
        params: dict[str, str] = {
            "apiKey": self._api_key,
            "limit": "50000",
            "sort": "asc",
        }

        async with aiohttp.ClientSession() as session, self._semaphore:
            try:
                timeout = aiohttp.ClientTimeout(total=30)
                async with session.get(url, params=params, timeout=timeout) as resp:
                    if resp.status == 403:
                        logger.error(
                            "Polygon returned 403 for %s — check API key and subscription tier.",
                            symbol,
                        )
                        return pd.DataFrame()
                    if resp.status != 200:
                        logger.error(
                            "Polygon returned HTTP %d for %s.", resp.status, symbol
                        )
                        return pd.DataFrame()
                    data: dict[str, Any] = await resp.json()
            except Exception as exc:
                logger.error("Network error fetching bars for %s: %s", symbol, exc)
                return pd.DataFrame()

        results = data.get("results", [])
        if not results:
            logger.warning("Polygon returned zero bars for %s [%s – %s].", symbol, start, end)
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.rename(
            columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            }
        )
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        df["symbol"] = symbol
        df = df[["date", "symbol", "open", "high", "low", "close", "volume"]].copy()
        df = df.sort_values("date").reset_index(drop=True)

        if use_cache:
            df.to_parquet(cache_file, index=False)
            logger.info("Cached %d daily bars for %s -> %s", len(df), symbol, cache_file)

        return df

    async def get_universe_bars(
        self,
        symbols: list[str],
        start: str,
        end: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch daily bars for a list of symbols concurrently.

        Runs all symbol fetches in parallel, bounded by the rate-limiter
        semaphore (default 5 concurrent requests).

        Args:
            symbols: List of equity tickers.
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            use_cache: Passed through to each ``get_daily_bars`` call.

        Returns:
            Combined DataFrame for all symbols, sorted by ``[date, symbol]``.
            Symbols that return errors are silently dropped.
        """
        tasks = [
            self.get_daily_bars(sym, start, end, use_cache=use_cache)
            for sym in symbols
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        dfs: list[pd.DataFrame] = []
        for sym, result in zip(symbols, raw_results, strict=False):
            if isinstance(result, Exception):
                logger.warning("Failed to fetch bars for %s: %s", sym, result)
            elif isinstance(result, pd.DataFrame) and not result.empty:
                dfs.append(result)

        if not dfs:
            logger.error("No data returned for any symbol in universe.")
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)
        logger.info(
            "Universe bars loaded: %d symbols, %d rows, %s – %s",
            combined["symbol"].nunique(),
            len(combined),
            combined["date"].min(),
            combined["date"].max(),
        )
        return combined

    async def get_options_iv(
        self,
        symbol: str,
        start: str,
        end: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch historical ATM 30-day implied volatility for a symbol.

        Polygon's free tier has limited options history. A paid subscription
        (Starter or higher) is required for full historical options data back
        to 2015. For higher-quality IV history consider ORATS, IVolatility,
        or Cboe DataShop.

        Args:
            symbol: Equity ticker.
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            use_cache: Use local Parquet cache when available.

        Returns:
            DataFrame with columns ``[date, symbol, iv_30]``, or an empty
            DataFrame when historical IV data is unavailable.
        """
        cache_file = self._cache_dir / f"iv_{symbol}_{start}_{end}.parquet"

        if use_cache and cache_file.exists():
            logger.debug("Cache hit — loading IV for %s from %s", symbol, cache_file)
            return pd.read_parquet(cache_file)

        # Polygon does not provide a clean historical IV surface endpoint on
        # free or starter tiers. On eligible plans, options chain snapshots can
        # be reconstructed, but doing so requires fetching chains for every
        # expiry date — expensive in both API calls and storage.
        #
        # For backtesting with the HV-IV backtester, the recommended fallback
        # is to let HVIVBacktester use the HV*1.15 proxy and note that this
        # UNDERSTATES signal quality (real IV is noisier and mean-reverts more
        # strongly than the proxy implies).
        logger.warning(
            "Historical IV for %s is not available on the free Polygon tier. "
            "Returning empty DataFrame — the HV-IV backtester will fall back "
            "to an HV * 1.15 proxy. For accurate signal validation use a paid "
            "IV data source (ORATS, IVolatility, or Cboe DataShop).",
            symbol,
        )
        return pd.DataFrame()

    async def get_universe_iv(
        self,
        symbols: list[str],
        start: str,
        end: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch historical IV for a list of symbols concurrently.

        Args:
            symbols: List of equity tickers.
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            use_cache: Use local Parquet cache when available.

        Returns:
            Combined IV DataFrame or empty DataFrame when unavailable.
        """
        tasks = [
            self.get_options_iv(sym, start, end, use_cache=use_cache)
            for sym in symbols
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        dfs: list[pd.DataFrame] = [
            r
            for r in raw_results
            if isinstance(r, pd.DataFrame) and not r.empty
        ]

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        return combined.sort_values(["date", "symbol"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Synchronous convenience wrappers
    # ------------------------------------------------------------------

    def fetch_daily_bars_sync(
        self,
        symbol: str,
        start: str,
        end: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Synchronous wrapper around ``get_daily_bars``.

        Suitable for use in scripts and notebooks that are not running
        inside an existing event loop.

        Args:
            symbol: Equity ticker.
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            use_cache: Use local Parquet cache when available.

        Returns:
            Daily OHLCV DataFrame (see ``get_daily_bars``).
        """
        return asyncio.run(self.get_daily_bars(symbol, start, end, use_cache))

    def fetch_universe_bars_sync(
        self,
        symbols: list[str],
        start: str,
        end: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Synchronous wrapper around ``get_universe_bars``.

        Args:
            symbols: List of equity tickers.
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            use_cache: Use local Parquet cache when available.

        Returns:
            Combined daily OHLCV DataFrame (see ``get_universe_bars``).
        """
        return asyncio.run(
            self.get_universe_bars(symbols, start, end, use_cache)
        )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self, symbol: str | None = None) -> int:
        """Delete cached Parquet files.

        Args:
            symbol: When provided, only files whose names contain this
                    ticker are removed. When None, all cache files are
                    removed.

        Returns:
            Number of files deleted.
        """
        pattern = f"*{symbol}*" if symbol else "*.parquet"
        files = list(self._cache_dir.glob(pattern))
        for f in files:
            f.unlink()
        logger.info("Cleared %d cache file(s) from %s", len(files), self._cache_dir)
        return len(files)

    def cache_stats(self) -> dict[str, Any]:
        """Return a summary of the local cache.

        Returns:
            Dictionary with ``file_count``, ``total_size_mb``, and
            ``cache_dir`` keys.
        """
        parquet_files = list(self._cache_dir.glob("*.parquet"))
        total_bytes = sum(f.stat().st_size for f in parquet_files)
        return {
            "file_count": len(parquet_files),
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
            "cache_dir": str(self._cache_dir),
        }


# ---------------------------------------------------------------------------
# CLI entry point (quick smoke-test / data pre-fetch)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Pre-fetch and cache historical bars from Polygon.io"
    )
    parser.add_argument(
        "--symbols",
        default="AAPL,MSFT,GOOGL",
        help="Comma-separated tickers (default: AAPL,MSFT,GOOGL)",
    )
    parser.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-01-01", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--sp500-liquid-50",
        action="store_true",
        help="Fetch the full SP500_LIQUID_50 universe (ignores --symbols)",
    )
    parser.add_argument(
        "--cache-stats",
        action="store_true",
        help="Print cache statistics and exit",
    )
    args = parser.parse_args()

    try:
        provider = PolygonDataProvider()
    except (ValueError, ImportError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if args.cache_stats:
        stats = provider.cache_stats()
        print(f"Cache directory : {stats['cache_dir']}")
        print(f"Files           : {stats['file_count']}")
        print(f"Total size      : {stats['total_size_mb']} MB")
        sys.exit(0)

    symbols = SP500_LIQUID_50 if args.sp500_liquid_50 else [
        s.strip() for s in args.symbols.split(",") if s.strip()
    ]

    print(f"Fetching daily bars for {len(symbols)} symbol(s) [{args.start} – {args.end}]")
    df = asyncio.run(
        provider.get_universe_bars(symbols, args.start, args.end)
    )

    if df.empty:
        print("No data returned. Verify POLYGON_API_KEY and subscription tier.")
        sys.exit(1)

    print(f"\nLoaded {len(df):,} rows across {df['symbol'].nunique()} symbols.")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nSample (last 5 rows):\n{df.tail().to_string(index=False)}")
    print(f"\nCache stats: {provider.cache_stats()}")
