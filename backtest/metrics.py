"""Performance metrics for options backtesting.

All functions accept pandas Series or DataFrames and return scalar floats,
making them easy to compose into a full metrics report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(
    returns: pd.Series,
    risk_free: float = 0.05,
    periods: int = 252,
) -> float:
    """Annualised Sharpe ratio.

    Args:
        returns: Periodic returns (e.g., daily P&L / prior equity).
        risk_free: Annual risk-free rate (default 5%).
        periods: Number of periods per year used for annualisation.

    Returns:
        Annualised Sharpe ratio, or 0.0 if standard deviation is zero.
    """
    if returns.empty or returns.std() == 0:
        return 0.0
    excess = returns - risk_free / periods
    return float(excess.mean() / excess.std() * np.sqrt(periods))


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.05,
    periods: int = 252,
) -> float:
    """Annualised Sortino ratio (penalises downside deviation only).

    Args:
        returns: Periodic returns.
        risk_free: Annual risk-free rate.
        periods: Number of periods per year.

    Returns:
        Annualised Sortino ratio, or 0.0 if downside deviation is zero.
    """
    if returns.empty:
        return 0.0
    excess = returns - risk_free / periods
    downside = excess[excess < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(periods))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative decimal fraction.

    Args:
        equity_curve: Equity values over time (must be positive).

    Returns:
        Max drawdown as a negative fraction (e.g., -0.25 means -25%).
        Returns 0.0 if the equity curve is empty or flat.
    """
    if equity_curve.empty:
        return 0.0
    rolling_max = equity_curve.cummax()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    return float(drawdowns.min())


def win_rate(trades: pd.DataFrame) -> float:
    """Fraction of trades that closed with a positive P&L.

    Args:
        trades: DataFrame with at least a ``pnl`` column.

    Returns:
        Win rate in [0, 1]. Returns 0.0 for an empty trade log.
    """
    if trades.empty or "pnl" not in trades.columns:
        return 0.0
    winners = trades["pnl"] > 0
    return float(winners.sum() / len(trades))


def profit_factor(trades: pd.DataFrame) -> float:
    """Ratio of gross profit to gross loss across all trades.

    Args:
        trades: DataFrame with a ``pnl`` column.

    Returns:
        Profit factor (>1 is net profitable). Returns 0.0 if there are no
        losing trades; returns ``inf`` if there are only winning trades.
    """
    if trades.empty or "pnl" not in trades.columns:
        return 0.0
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = trades.loc[trades["pnl"] < 0, "pnl"].abs().sum()
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def calmar_ratio(returns: pd.Series, equity_curve: pd.Series) -> float:
    """Calmar ratio: annualised return divided by absolute max drawdown.

    Args:
        returns: Periodic returns.
        equity_curve: Equity values over the same period.

    Returns:
        Calmar ratio. Returns 0.0 if max drawdown is zero.
    """
    if returns.empty or equity_curve.empty:
        return 0.0
    annual_return = float(returns.mean() * 252)
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return annual_return / mdd


def compute_all(
    trades: pd.DataFrame,
    equity_curve: pd.Series,
    returns: pd.Series,
) -> dict[str, float]:
    """Compute all metrics and return them as a single dictionary.

    Args:
        trades: Closed-trade log with a ``pnl`` column.
        equity_curve: Running equity values.
        returns: Periodic returns derived from the equity curve.

    Returns:
        Dictionary mapping metric names to their scalar values.
    """
    return {
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity_curve),
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
        "calmar_ratio": calmar_ratio(returns, equity_curve),
        "total_trades": len(trades),
        "total_pnl": float(trades["pnl"].sum()) if not trades.empty else 0.0,
        "avg_pnl_per_trade": float(trades["pnl"].mean()) if not trades.empty else 0.0,
        "final_equity": float(equity_curve.iloc[-1]) if not equity_curve.empty else 0.0,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    rets = pd.Series(rng.normal(0.001, 0.02, 252))
    equity = (1 + rets).cumprod() * 10_000
    pnl_values = rng.normal(50, 200, 30)
    trade_log = pd.DataFrame({"pnl": pnl_values})

    results = compute_all(trade_log, equity, rets)
    print("Metrics:")
    for k, v in results.items():
        print(f"  {k:<25}: {v:.4f}")
