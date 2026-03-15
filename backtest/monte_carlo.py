"""Monte Carlo simulation via bootstrapped trade resampling.

Resamples the closed-trade log with replacement to build a distribution of
equity curves and derive confidence intervals on key risk metrics.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)


def _build_equity_curve(pnl_sequence: np.ndarray, initial_capital: float) -> np.ndarray:
    """Build a running equity curve from a sequence of trade P&Ls.

    Args:
        pnl_sequence: Array of per-trade net P&L values.
        initial_capital: Starting capital in dollars.

    Returns:
        Equity array of length ``len(pnl_sequence) + 1`` starting at
        ``initial_capital``.
    """
    equity = np.empty(len(pnl_sequence) + 1)
    equity[0] = initial_capital
    np.cumsum(pnl_sequence, out=equity[1:])
    equity[1:] += initial_capital
    return equity


def _max_drawdown_array(equity: np.ndarray) -> float:
    """Fast max drawdown calculation over a NumPy equity array.

    Args:
        equity: 1-D array of equity values (must be positive).

    Returns:
        Max drawdown as a negative fraction. Returns 0.0 for flat arrays.
    """
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak == 0, 1.0, peak)
    return float(dd.min())


class MonteCarloSimulator:
    """Bootstrap Monte Carlo simulator for options backtest trade logs.

    Resamples completed trades (with replacement) to estimate the
    distribution of outcomes that could arise from a strategy with the
    same statistical properties.

    Args:
        random_seed: Optional integer seed for reproducible results.
    """

    def __init__(self, random_seed: int | None = None) -> None:
        self._rng = np.random.default_rng(random_seed)

    def run(
        self,
        trades: pd.DataFrame,
        n_simulations: int = 10_000,
        initial_capital: float = 10_000.0,
    ) -> dict:
        """Run the Monte Carlo simulation.

        Each simulation randomly resamples ``len(trades)`` trades with
        replacement from the historical trade log and computes equity curve
        statistics for that path.

        Args:
            trades: Closed-trade DataFrame with at least a ``pnl`` column.
            n_simulations: Number of bootstrap paths to generate.
            initial_capital: Starting capital for each simulated path.

        Returns:
            Dictionary containing:
                - ``mean_return``: Mean final equity across simulations.
                - ``median_return``: Median final equity.
                - ``p5_return``: 5th-percentile final equity.
                - ``p95_return``: 95th-percentile final equity.
                - ``max_drawdown_p5``: 5th-percentile max drawdown.
                - ``max_drawdown_p50``: Median max drawdown.
                - ``max_drawdown_p95``: 95th-percentile max drawdown.
                - ``prob_ruin``: Fraction of paths where equity fell below
                  50% of initial capital at any point.
                - ``n_simulations``: Number of paths run.
                - ``n_trades``: Number of trades in the input log.
        """
        if trades.empty or "pnl" not in trades.columns:
            logger.warning("Empty trade log — returning zero-fill results.")
            return self._empty_result(initial_capital, n_simulations)

        pnl_values = trades["pnl"].to_numpy(dtype=float)
        n_trades = len(pnl_values)
        ruin_threshold = initial_capital * 0.50

        final_equities = np.empty(n_simulations)
        drawdowns = np.empty(n_simulations)
        ruin_count = 0

        for sim in range(n_simulations):
            sampled = self._rng.choice(pnl_values, size=n_trades, replace=True)
            equity = _build_equity_curve(sampled, initial_capital)
            final_equities[sim] = equity[-1]
            drawdowns[sim] = _max_drawdown_array(equity)
            if np.any(equity <= ruin_threshold):
                ruin_count += 1

        return {
            "mean_return": float(np.mean(final_equities)),
            "median_return": float(np.median(final_equities)),
            "p5_return": float(np.percentile(final_equities, 5)),
            "p95_return": float(np.percentile(final_equities, 95)),
            "max_drawdown_p5": float(np.percentile(drawdowns, 5)),
            "max_drawdown_p50": float(np.percentile(drawdowns, 50)),
            "max_drawdown_p95": float(np.percentile(drawdowns, 95)),
            "prob_ruin": float(ruin_count / n_simulations),
            "n_simulations": n_simulations,
            "n_trades": n_trades,
            "initial_capital": initial_capital,
        }

    @staticmethod
    def _empty_result(initial_capital: float, n_simulations: int) -> dict:
        """Return a zero-filled result dictionary for empty trade logs."""
        return {
            "mean_return": initial_capital,
            "median_return": initial_capital,
            "p5_return": initial_capital,
            "p95_return": initial_capital,
            "max_drawdown_p5": 0.0,
            "max_drawdown_p50": 0.0,
            "max_drawdown_p95": 0.0,
            "prob_ruin": 0.0,
            "n_simulations": n_simulations,
            "n_trades": 0,
            "initial_capital": initial_capital,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Monte Carlo simulation on backtest trades")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-03-01")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--n-sims", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from data_loader import PolygonDataLoader
    from replay_engine import ReplayEngine

    loader = PolygonDataLoader()
    df = loader.load_stock_bars(args.symbol, args.start, args.end, "day")
    if df.empty:
        print("No data loaded; exiting.")
        sys.exit(1)

    engine = ReplayEngine()
    result = engine.run(df, initial_capital=args.capital)

    if result.trades.empty:
        print("No trades generated by backtest — cannot run Monte Carlo.")
        sys.exit(0)

    print(f"Running {args.n_sims:,} Monte Carlo simulations on {len(result.trades)} trades...")
    simulator = MonteCarloSimulator(random_seed=args.seed)
    mc_results = simulator.run(result.trades, n_simulations=args.n_sims, initial_capital=args.capital)

    print("\nMonte Carlo Results")
    print("=" * 50)
    for key, val in mc_results.items():
        if isinstance(val, float):
            print(f"  {key:<28}: {val:>12.4f}")
        else:
            print(f"  {key:<28}: {val:>12}")
