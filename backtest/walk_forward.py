"""Walk-forward optimisation for the contrarian options strategy.

Splits the full date range into rolling in-sample (train) and out-of-sample
(test) windows, fits the best RSI/Bollinger Band period combination on the
train window, and evaluates it on the test window.  Results across all
windows are aggregated into a single DataFrame for analysis.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta  # type: ignore[import]

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import PolygonDataLoader
from replay_engine import DEFAULT_CONFIG, ReplayEngine

logger = logging.getLogger(__name__)


def _date_windows(
    start: str,
    end: str,
    train_months: int,
    test_months: int,
) -> list[tuple[str, str, str, str]]:
    """Generate rolling (train_start, train_end, test_start, test_end) tuples.

    Args:
        start: Overall start date as ``"YYYY-MM-DD"``.
        end: Overall end date as ``"YYYY-MM-DD"``.
        train_months: Number of months in each in-sample window.
        test_months: Number of months in each out-of-sample window.

    Returns:
        List of 4-tuples ``(train_start, train_end, test_start, test_end)``
        as ISO date strings. Returns an empty list if the range is too short.
    """
    windows: list[tuple[str, str, str, str]] = []
    cursor = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    while True:
        train_start = cursor
        train_end = cursor + relativedelta(months=train_months)
        test_start = train_end
        test_end = test_start + relativedelta(months=test_months)
        if test_end > end_ts:
            break
        windows.append((
            train_start.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d"),
            test_start.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d"),
        ))
        cursor += relativedelta(months=test_months)  # step forward by test period

    return windows


def _score(metrics: dict) -> float:
    """Composite in-sample score balancing return, risk, and consistency.

    Args:
        metrics: Output of ``compute_all``.

    Returns:
        Scalar score (higher is better).
    """
    sharpe = metrics.get("sharpe_ratio", 0.0)
    pf = min(metrics.get("profit_factor", 0.0), 5.0)  # cap to avoid outlier domination
    wr = metrics.get("win_rate", 0.0)
    n = max(metrics.get("total_trades", 0), 1)
    trade_bonus = np.log1p(n) * 0.1
    return sharpe + 0.5 * pf + wr + trade_bonus


class WalkForwardOptimizer:
    """Rolling walk-forward optimiser over RSI and Bollinger Band parameters.

    Args:
        train_months: In-sample window length in months.
        test_months: Out-of-sample window length in months.
    """

    def __init__(self, train_months: int = 6, test_months: int = 2) -> None:
        self.train_months = train_months
        self.test_months = test_months
        self._loader = PolygonDataLoader()

    def run(
        self,
        symbols: list[str],
        start: str,
        end: str,
        param_grid: dict[str, list[Any]],
    ) -> pd.DataFrame:
        """Execute the full walk-forward optimisation.

        For each symbol and each rolling window the method:
          1. Loads (or generates) bar data.
          2. Iterates over every parameter combination in ``param_grid``.
          3. Selects the best combination on in-sample data using a composite score.
          4. Evaluates the selected parameters on the out-of-sample window.

        Args:
            symbols: List of ticker symbols to optimise.
            start: Start of the full optimisation period (``"YYYY-MM-DD"``).
            end: End of the full optimisation period (``"YYYY-MM-DD"``).
            param_grid: Dictionary mapping parameter names to lists of values.
                        Supported keys: ``rsi_period``, ``bb_period``,
                        ``rsi_oversold``, ``volume_spike_mult``.

        Returns:
            DataFrame with one row per (symbol, window, best_params) combination,
            containing both in-sample and out-of-sample metrics.
        """
        windows = _date_windows(start, end, self.train_months, self.test_months)
        if not windows:
            logger.warning("Date range too short to build any walk-forward windows.")
            return pd.DataFrame()

        # Build the Cartesian product of the parameter grid once
        param_keys = list(param_grid.keys())
        param_combos = [
            dict(zip(param_keys, vals, strict=False))
            for vals in itertools.product(*[param_grid[k] for k in param_keys])
        ]
        logger.info(
            "Walk-forward: %d symbols × %d windows × %d param combos = %d runs",
            len(symbols),
            len(windows),
            len(param_combos),
            len(symbols) * len(windows) * len(param_combos),
        )

        records: list[dict] = []

        for symbol in symbols:
            logger.info("Loading full bar data for %s [%s – %s]", symbol, start, end)
            full_df = self._loader.load_stock_bars(symbol, start, end, timespan="day")
            if full_df.empty:
                logger.warning("No data for %s, skipping.", symbol)
                continue

            for win_idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
                train_df = full_df[
                    (full_df["timestamp"] >= tr_s) & (full_df["timestamp"] < tr_e)
                ].reset_index(drop=True)
                test_df = full_df[
                    (full_df["timestamp"] >= te_s) & (full_df["timestamp"] < te_e)
                ].reset_index(drop=True)

                if len(train_df) < 30 or len(test_df) < 5:
                    logger.debug(
                        "Window %d: insufficient bars (train=%d, test=%d).",
                        win_idx,
                        len(train_df),
                        len(test_df),
                    )
                    continue

                # --- In-sample grid search -----------------------------------
                best_score = -np.inf
                best_params: dict = param_combos[0]
                best_is_metrics: dict = {}

                for params in param_combos:
                    cfg = {**DEFAULT_CONFIG, **params}
                    try:
                        engine = ReplayEngine(config=cfg)
                        result = engine.run(train_df, initial_capital=10_000.0)
                        score = _score(result.metrics)
                        if score > best_score:
                            best_score = score
                            best_params = params
                            best_is_metrics = result.metrics
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("IS eval error (params=%s): %s", params, exc)
                        continue

                # --- Out-of-sample evaluation --------------------------------
                oos_metrics: dict = {}
                try:
                    cfg_oos = {**DEFAULT_CONFIG, **best_params}
                    engine_oos = ReplayEngine(config=cfg_oos)
                    oos_result = engine_oos.run(test_df, initial_capital=10_000.0)
                    oos_metrics = oos_result.metrics
                except Exception as exc:  # noqa: BLE001
                    logger.warning("OOS eval error for %s window %d: %s", symbol, win_idx, exc)

                record: dict = {
                    "symbol": symbol,
                    "window": win_idx,
                    "train_start": tr_s,
                    "train_end": tr_e,
                    "test_start": te_s,
                    "test_end": te_e,
                    "is_score": round(best_score, 4),
                }
                for k, v in best_params.items():
                    record[f"param_{k}"] = v
                for k, v in best_is_metrics.items():
                    record[f"is_{k}"] = v
                for k, v in oos_metrics.items():
                    record[f"oos_{k}"] = v

                records.append(record)
                logger.info(
                    "  Window %d (%s–%s): best params=%s  IS_score=%.3f  "
                    "OOS_sharpe=%.3f  OOS_trades=%d",
                    win_idx,
                    te_s,
                    te_e,
                    best_params,
                    best_score,
                    oos_metrics.get("sharpe_ratio", float("nan")),
                    int(oos_metrics.get("total_trades", 0)),
                )

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Walk-forward optimisation")
    parser.add_argument(
        "--symbols", default="AAPL,MSFT,TSLA",
        help="Comma-separated list of tickers",
    )
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-03-01", help="End date YYYY-MM-DD")
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--test-months", type=int, default=2)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    param_grid: dict[str, list[Any]] = {
        "rsi_period": [10, 14, 20],
        "bb_period": [15, 20, 25],
        "rsi_oversold": [25, 30, 35],
        "volume_spike_mult": [2.0, 2.5, 3.0],
    }

    optimizer = WalkForwardOptimizer(
        train_months=args.train_months,
        test_months=args.test_months,
    )
    results_df = optimizer.run(symbols, args.start, args.end, param_grid)

    if results_df.empty:
        print("No results generated. Check date range and data availability.")
        sys.exit(0)

    print("\nWalk-Forward Summary")
    print("=" * 70)
    oos_cols = [c for c in results_df.columns if c.startswith("oos_")]
    display_cols = ["symbol", "window", "test_start", "test_end", "is_score"] + oos_cols[:6]
    display_cols = [c for c in display_cols if c in results_df.columns]
    print(results_df[display_cols].to_string(index=False))

    # Aggregate OOS stats
    sharpe_col = "oos_sharpe_ratio"
    if sharpe_col in results_df.columns:
        print(f"\nMean OOS Sharpe: {results_df[sharpe_col].mean():.4f}")
        print(f"Median OOS Sharpe: {results_df[sharpe_col].median():.4f}")
