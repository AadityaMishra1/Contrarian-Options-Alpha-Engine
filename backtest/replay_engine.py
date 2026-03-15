"""Row-by-row event replay engine for the contrarian options strategy.

Entry conditions (all must be true):
  - RSI < ``rsi_oversold`` (default 30)
  - Close price < lower Bollinger Band
  - Volume spike: current volume > ``volume_spike_mult`` * rolling mean volume

Exit conditions (first triggered wins):
  - RSI crosses above ``rsi_exit`` (default 50)
  - DTE reaches zero (option expiry)
  - Stop-loss: position P&L <= ``stop_loss_pct`` * entry cost (default -50%)

Position sizing uses half-Kelly based on estimated win probability and
average win/loss ratio derived from a rolling look-back window.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from cost_model import CostModel
from metrics import compute_all

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default strategy configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict = {
    "rsi_period": 14,
    "bb_period": 20,
    "bb_std": 2.0,
    "rsi_oversold": 30,
    "rsi_exit": 50,
    "volume_spike_mult": 2.5,
    "volume_spike_window": 20,
    "option_min_price": 0.05,
    "option_max_price": 0.30,
    "option_dte_min": 1,
    "option_dte_max": 5,
    "stop_loss_pct": -0.50,
    "kelly_fraction": 0.5,
    "max_position_pct": 0.10,  # max 10% of capital per trade
    "commission_per_contract": 0.65,
    "slippage_ticks": 1,
    "tick_size": 0.01,
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    """Container for all backtest outputs.

    Attributes:
        trades: Closed-trade log with columns [entry_time, exit_time,
            entry_price, exit_price, contracts, pnl, entry_cost,
            exit_signal, option_type].
        equity_curve: Running portfolio equity indexed by bar timestamp.
        metrics: Dictionary of computed performance metrics.
        params: Strategy configuration used for the run.
    """

    trades: pd.DataFrame
    equity_curve: pd.Series
    metrics: dict
    params: dict


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI computed with exponential smoothing.

    Args:
        close: Closing price series.
        period: Look-back period (typically 14).

    Returns:
        RSI series in [0, 100], NaN for the first ``period`` bars.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _bollinger_bands(
    close: pd.Series, period: int, num_std: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Simple moving average Bollinger Bands.

    Args:
        close: Closing price series.
        period: Rolling window length.
        num_std: Number of standard deviations for the bands.

    Returns:
        Tuple of (upper_band, middle_band, lower_band) as Series.
    """
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + num_std * std, mid, mid - num_std * std


def _volume_spike(volume: pd.Series, window: int, multiplier: float) -> pd.Series:
    """Boolean Series indicating a volume spike vs. rolling mean.

    Args:
        volume: Volume series.
        window: Look-back window for rolling mean.
        multiplier: Spike threshold multiplier.

    Returns:
        Boolean Series: True where volume exceeds ``multiplier * mean``.
    """
    rolling_mean = volume.rolling(window).mean()
    return volume > (multiplier * rolling_mean)


def _compute_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Attach all strategy indicators to a bar DataFrame in-place copy.

    Args:
        df: OHLCV DataFrame with at least [close, volume] columns.
        config: Strategy configuration dictionary.

    Returns:
        New DataFrame with indicator columns appended.
    """
    out = df.copy()
    out["rsi"] = _rsi(out["close"], config["rsi_period"])
    upper, mid, lower = _bollinger_bands(
        out["close"], config["bb_period"], config["bb_std"]
    )
    out["bb_upper"] = upper
    out["bb_mid"] = mid
    out["bb_lower"] = lower
    out["volume_spike"] = _volume_spike(
        out["volume"], config["volume_spike_window"], config["volume_spike_mult"]
    )
    return out


# ---------------------------------------------------------------------------
# Synthetic option pricer
# ---------------------------------------------------------------------------

def _synthetic_option_price(
    spot: float,
    moneyness_otm: float = 0.03,
    dte: int = 3,
    iv: float = 0.35,
) -> tuple[float, float]:
    """Very rough proxy for an OTM option mid price and spread.

    Uses a simplified log-normal approximation: price decays with both
    moneyness distance and time-to-expiry. Suitable for backtesting when
    a live options chain is not available.

    Args:
        spot: Underlying spot price.
        moneyness_otm: Distance from ATM as a fraction of spot (e.g. 0.03 = 3%).
        dte: Days-to-expiry of the option.
        iv: Annualised implied volatility (e.g. 0.35 = 35%).

    Returns:
        Tuple of (mid_price, spread) in dollars per share.
    """
    t = dte / 365.0
    # Rough BS-like time value proxy
    time_value = spot * iv * np.sqrt(t) * 0.4
    distance_discount = np.exp(-moneyness_otm / (iv * np.sqrt(t) + 1e-9))
    mid = max(0.01, time_value * distance_discount)
    spread = max(0.01, mid * 0.20)  # 20% spread typical for cheap OTM options
    return round(mid, 2), round(spread, 2)


# ---------------------------------------------------------------------------
# Half-Kelly position sizer
# ---------------------------------------------------------------------------

def _half_kelly_contracts(
    capital: float,
    option_cost: float,
    win_prob: float,
    avg_win: float,
    avg_loss: float,
    max_position_pct: float,
) -> int:
    """Calculate position size using the half-Kelly criterion.

    Args:
        capital: Current portfolio equity.
        option_cost: Total cost to open 1 contract (premium * 100 + costs).
        win_prob: Estimated probability of a winning trade (0–1).
        avg_win: Average gain on a winning trade (dollars, positive).
        avg_loss: Average loss on a losing trade (dollars, positive magnitude).
        max_position_pct: Maximum fraction of capital to risk on one trade.

    Returns:
        Number of contracts to trade (minimum 1).
    """
    if avg_loss <= 0 or option_cost <= 0:
        return 1
    b = avg_win / avg_loss  # odds ratio
    p = max(0.01, min(0.99, win_prob))
    q = 1 - p
    kelly = (b * p - q) / b
    half_kelly = max(0.0, kelly / 2)
    max_risk = capital * max_position_pct
    contracts = int(max_risk * half_kelly / option_cost)
    return max(1, contracts)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class ReplayEngine:
    """Bar-by-bar backtest replay engine for the contrarian options strategy.

    Args:
        config: Strategy parameter dictionary. Any missing keys fall back to
                ``DEFAULT_CONFIG``.
    """

    def __init__(self, config: dict | None = None) -> None:
        self.config: dict = {**DEFAULT_CONFIG, **(config or {})}
        self.cost_model = CostModel(
            commission_per_contract=self.config["commission_per_contract"],
            slippage_ticks=self.config["slippage_ticks"],
            tick_size=self.config["tick_size"],
        )

    def run(
        self,
        df: pd.DataFrame,
        initial_capital: float = 10_000.0,
    ) -> BacktestResult:
        """Execute the full backtest over the supplied bar data.

        Args:
            df: OHLCV DataFrame with columns [timestamp, open, high, low,
                close, volume]. The ``timestamp`` column should be
                datetime-like.
            initial_capital: Starting portfolio equity in dollars.

        Returns:
            ``BacktestResult`` with trades, equity curve, and all metrics.
        """
        cfg = self.config
        df = _compute_indicators(df.copy(), cfg)
        df = df.dropna(subset=["rsi", "bb_lower"]).reset_index(drop=True)

        capital = initial_capital
        trade_log: list[dict] = []
        equity_points: list[tuple] = []

        # Rolling stats for Kelly sizing
        recent_wins: list[float] = []
        recent_losses: list[float] = []

        # Active position state
        in_position = False
        entry_idx: int = 0
        entry_price: float = 0.0
        entry_spread: float = 0.0
        entry_cost_total: float = 0.0
        contracts: int = 1
        entry_dte: int = 3
        option_type: str = "put"

        for i, row in df.iterrows():
            ts = row.get("timestamp", i)
            close = float(row["close"])
            rsi = float(row["rsi"])
            bb_lower = float(row["bb_lower"])
            vol_spike = bool(row["volume_spike"])

            # ---- Exit logic -----------------------------------------------
            if in_position:
                bars_held = i - entry_idx  # type: ignore[operator]
                current_mid, current_spread = _synthetic_option_price(
                    close,
                    moneyness_otm=0.03,
                    dte=max(1, entry_dte - bars_held // max(1, len(df) // entry_dte)),
                )
                position_value = contracts * current_mid * 100
                initial_value = contracts * entry_price * 100
                pnl_pct = (position_value - initial_value) / max(1.0, initial_value)

                exit_signal = None
                if rsi > cfg["rsi_exit"]:
                    exit_signal = "rsi_cross"
                elif bars_held >= entry_dte:
                    exit_signal = "dte_expire"
                    current_mid = 0.0  # option expired worthless
                elif pnl_pct <= cfg["stop_loss_pct"]:
                    exit_signal = "stop_loss"

                if exit_signal:
                    exit_cost = self.cost_model.total_cost(contracts, current_mid, current_spread)
                    gross_pnl = contracts * (current_mid - entry_price) * 100
                    net_pnl = gross_pnl - entry_cost_total - exit_cost
                    capital += net_pnl

                    trade_log.append(
                        {
                            "entry_time": df.iloc[entry_idx]["timestamp"] if "timestamp" in df.columns else entry_idx,  # type: ignore[call-overload]
                            "exit_time": ts,
                            "entry_price": entry_price,
                            "exit_price": current_mid,
                            "contracts": contracts,
                            "pnl": round(net_pnl, 4),
                            "entry_cost": round(entry_cost_total, 4),
                            "exit_signal": exit_signal,
                            "option_type": option_type,
                        }
                    )

                    if net_pnl > 0:
                        recent_wins.append(net_pnl)
                    else:
                        recent_losses.append(abs(net_pnl))

                    # Keep rolling windows bounded
                    recent_wins = recent_wins[-50:]
                    recent_losses = recent_losses[-50:]

                    in_position = False

            # ---- Entry logic -----------------------------------------------
            if (
                not in_position
                and rsi < cfg["rsi_oversold"]
                and close < bb_lower
                and vol_spike
                and capital > 100  # guard against near-ruin
            ):
                # Determine option type: OTM put for bearish mean reversion
                option_type = "put"
                entry_dte = int(np.random.randint(cfg["option_dte_min"], cfg["option_dte_max"] + 1))
                entry_price, entry_spread = _synthetic_option_price(
                    close, moneyness_otm=0.03, dte=entry_dte
                )

                # Filter: only trade options within price range
                if not (cfg["option_min_price"] <= entry_price <= cfg["option_max_price"]):
                    equity_points.append((ts, capital))
                    continue

                # Half-Kelly sizing
                win_prob = len(recent_wins) / max(1, len(recent_wins) + len(recent_losses))
                avg_win = np.mean(recent_wins) if recent_wins else entry_price * 100
                avg_loss = np.mean(recent_losses) if recent_losses else entry_price * 50
                cost_per_contract = (
                    entry_price * 100
                    + self.cost_model.total_cost(1, entry_price, entry_spread)
                )
                contracts = _half_kelly_contracts(
                    capital,
                    cost_per_contract,
                    win_prob,
                    avg_win,
                    avg_loss,
                    cfg["max_position_pct"],
                )

                entry_cost_total = self.cost_model.total_cost(contracts, entry_price, entry_spread)
                total_debit = contracts * entry_price * 100 + entry_cost_total

                if total_debit > capital:
                    contracts = max(1, int(capital * cfg["max_position_pct"] / cost_per_contract))
                    entry_cost_total = self.cost_model.total_cost(contracts, entry_price, entry_spread)
                    total_debit = contracts * entry_price * 100 + entry_cost_total

                capital -= total_debit
                in_position = True
                entry_idx = int(i)  # type: ignore[arg-type]

            equity_points.append((ts, capital))

        # Close any open position at last bar mark-to-market
        if in_position and len(df) > 0:
            last_row = df.iloc[-1]
            last_close = float(last_row["close"])
            last_mid, last_spread = _synthetic_option_price(last_close, dte=1)
            exit_cost = self.cost_model.total_cost(contracts, last_mid, last_spread)
            gross_pnl = contracts * (last_mid - entry_price) * 100
            net_pnl = gross_pnl - entry_cost_total - exit_cost
            capital += net_pnl
            trade_log.append(
                {
                    "entry_time": df.iloc[entry_idx]["timestamp"] if "timestamp" in df.columns else entry_idx,
                    "exit_time": last_row.get("timestamp", len(df) - 1),
                    "entry_price": entry_price,
                    "exit_price": last_mid,
                    "contracts": contracts,
                    "pnl": round(net_pnl, 4),
                    "entry_cost": round(entry_cost_total, 4),
                    "exit_signal": "end_of_data",
                    "option_type": option_type,
                }
            )
            equity_points.append((equity_points[-1][0] if equity_points else 0, capital))

        # Build outputs
        trades_df = pd.DataFrame(trade_log)
        if not equity_points:
            equity_series = pd.Series([initial_capital], name="equity")
        else:
            timestamps, values = zip(*equity_points, strict=False)
            equity_series = pd.Series(values, index=timestamps, name="equity", dtype=float)

        returns = equity_series.pct_change().dropna()
        all_metrics = compute_all(trades_df, equity_series, returns)
        all_metrics["initial_capital"] = initial_capital
        all_metrics["final_capital"] = float(equity_series.iloc[-1])
        all_metrics["total_return_pct"] = (
            (float(equity_series.iloc[-1]) - initial_capital) / initial_capital * 100
        )

        return BacktestResult(
            trades=trades_df,
            equity_curve=equity_series,
            metrics=all_metrics,
            params=self.config,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_metrics(metrics: dict) -> None:
    """Pretty-print the metrics dictionary to stdout."""
    print("\n" + "=" * 55)
    print("  BACKTEST RESULTS")
    print("=" * 55)
    order = [
        "total_trades", "win_rate", "profit_factor", "total_return_pct",
        "sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown",
        "total_pnl", "avg_pnl_per_trade", "initial_capital", "final_capital",
    ]
    for key in order:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                print(f"  {key:<28}: {val:>10.4f}")
            else:
                print(f"  {key:<28}: {val:>10}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run contrarian options backtest")
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol")
    parser.add_argument("--start", default="2025-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-03-01", help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=10_000.0, help="Initial capital")
    parser.add_argument("--timespan", default="day", help="Bar granularity: minute/hour/day")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    args = parser.parse_args()

    # Load data
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import PolygonDataLoader

    loader = PolygonDataLoader()
    df = loader.load_stock_bars(args.symbol, args.start, args.end, args.timespan)

    if df.empty:
        print("ERROR: No data loaded. Exiting.")
        sys.exit(1)

    print(f"Loaded {len(df)} bars for {args.symbol} ({args.start} to {args.end})")

    # Run backtest
    engine = ReplayEngine()
    result = engine.run(df, initial_capital=args.capital)

    _print_metrics(result.metrics)

    if result.trades.empty:
        print("No trades were generated. Try adjusting strategy parameters.")
    else:
        print(f"Trades sample (last 5):\n{result.trades.tail().to_string()}\n")

    if args.report:
        from report import generate_report
        generate_report(result)
        print("HTML report generated in reports/")
