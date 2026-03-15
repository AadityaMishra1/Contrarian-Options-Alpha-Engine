"""HV-IV Gap Strategy Backtester.

Backtests the Goyal-Saretto (2009) HV-IV gap signal on historical
options data. Designed to run with real Polygon.io options history.

Usage:
    python -m backtest.hv_iv_backtest --start 2015-01-01 --end 2025-01-01
"""
from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IronCondorTrade:
    """Represents a single iron condor trade."""

    symbol: str
    entry_date: str
    exit_date: str = ""
    entry_credit: float = 0.0   # net credit after costs (dollars)
    exit_debit: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    max_risk: float = 0.0       # (wing_width - credit) * contracts * 100
    gap_ratio: float = 0.0      # IV/HV at entry
    regime: str = ""
    dte_at_entry: int = 0
    exit_reason: str = ""       # profit_target | stop_loss | time_exit | backtest_end
    is_winner: bool = False


@dataclass
class BacktestConfig:
    """Configuration for the HV-IV backtest."""

    start_date: str = "2015-01-01"
    end_date: str = "2025-01-01"
    initial_capital: float = 25_000.0

    # Signal parameters (Goyal-Saretto 2009)
    hv_window: int = 30         # days for realised-vol window
    min_gap_ratio: float = 1.2  # minimum IV/HV to enter
    universe_size: int = 50

    # Trade structure
    target_dte: int = 30        # entry DTE
    exit_dte: int = 14          # close at this DTE if still open
    wing_width: float = 5.0     # iron condor wing width ($)

    # Risk management
    profit_target_pct: float = 0.50    # close at 50% of max credit
    stop_loss_multiple: float = 2.0    # close if loss > 2x credit
    max_positions: int = 5
    max_position_pct: float = 0.03     # 3% of capital per trade

    # Costs (Muravyev & Pearson 2020 estimates)
    commission_per_contract: float = 0.65
    slippage_per_leg: float = 0.03     # $0.03 per leg

    output_dir: str = "backtest/results"


class HVIVBacktester:
    """Backtests the HV-IV gap iron condor strategy (Goyal-Saretto 2009).

    Processes historical data bar-by-bar, computes the IV/HV gap signal,
    enters iron condors on top-ranked candidates, and manages exits.

    Args:
        config: Strategy configuration. Uses BacktestConfig defaults when None.
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self.trades: list[IronCondorTrade] = []
        self.equity_curve: list[dict[str, Any]] = []
        self.capital = self.config.initial_capital
        self.open_positions: list[IronCondorTrade] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        price_data: pd.DataFrame,
        iv_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Run the backtest on historical price data.

        Args:
            price_data: DataFrame with [date, symbol, close, volume], sorted
                        ascending by date.
            iv_data: Optional DataFrame with [date, symbol, iv_30]. When None
                     the backtester uses HV * 1.15 as an IV proxy (mechanics
                     testing only — not valid for strategy research).

        Returns:
            Dict with ``"trades"``, ``"equity_curve"``, and ``"metrics"``.
        """
        dates = sorted(price_data["date"].unique())
        symbols = price_data["symbol"].unique().tolist()

        # Load VIX history for IV proxy
        self._load_vix()

        logger.info(
            "Starting HV-IV backtest: %s to %s, %d symbols, $%.0f capital",
            dates[0],
            dates[-1],
            len(symbols),
            self.capital,
        )

        for i, current_date in enumerate(dates):
            if str(current_date) < self.config.start_date:
                continue
            if str(current_date) > self.config.end_date:
                break

            # 1. Check exit conditions for all open positions
            self._check_exits(current_date, price_data)

            # 2. Weekly rebalance: scan for new entries
            if i % 5 == 0:
                self._scan_and_enter(current_date, price_data, iv_data, symbols)

            # 3. Record portfolio equity snapshot
            unrealized = sum(
                self._mark_to_market(pos, current_date, price_data)
                for pos in self.open_positions
            )
            self.equity_curve.append(
                {
                    "date": str(current_date),
                    "equity": self.capital + unrealized,
                    "capital": self.capital,
                    "open_positions": len(self.open_positions),
                }
            )

        # Force-close any remaining positions at end of data
        if dates:
            for pos in list(self.open_positions):
                self._close_position(pos, dates[-1], "backtest_end", 0.0)

        return self._compute_results()

    def _scan_and_enter(
        self,
        current_date: Any,
        price_data: pd.DataFrame,
        iv_data: pd.DataFrame | None,
        symbols: list[str],
    ) -> None:
        """Rank universe by IV/HV gap and enter top candidates."""
        if len(self.open_positions) >= self.config.max_positions:
            return

        candidates: list[tuple[str, float, float, float]] = []

        for symbol in symbols:
            sym_data = price_data[
                (price_data["symbol"] == symbol)
                & (price_data["date"] <= current_date)
            ].tail(self.config.hv_window + 5)

            if len(sym_data) < self.config.hv_window + 1:
                continue

            closes = sym_data["close"].values
            hv = self._realized_vol(closes, self.config.hv_window)

            if hv <= 0.01:
                continue

            if iv_data is not None:
                iv_row = iv_data[
                    (iv_data["symbol"] == symbol)
                    & (iv_data["date"] == current_date)
                ]
                iv = (
                    float(iv_row["iv_30"].values[0])
                    if len(iv_row) > 0
                    else hv * 1.15
                )
            else:
                # IV proxy using VIX + stock beta.
                # Stock IV ≈ beta * VIX_level (annualized).
                # VIX is loaded once and passed via iv_data="vix" convention.
                vix_today = self._get_vix(current_date)
                if vix_today > 0:
                    # Estimate stock beta from recent returns vs market
                    stock_vol_daily = hv / np.sqrt(252) if hv > 0 else 0.01
                    mkt_vol_daily = vix_today / (100 * np.sqrt(252))
                    beta_approx = stock_vol_daily / mkt_vol_daily if mkt_vol_daily > 0 else 1.0
                    beta_approx = max(0.5, min(3.0, beta_approx))  # clamp
                    iv = beta_approx * (vix_today / 100.0)
                else:
                    iv = hv * 1.15  # fallback

            gap_ratio = iv / hv if hv > 0 else 1.0
            if any(p.symbol == symbol for p in self.open_positions):
                continue

            if gap_ratio >= self.config.min_gap_ratio:
                candidates.append((symbol, gap_ratio, float(closes[-1]), iv))

        # Rank by gap_ratio descending, fill available slots
        candidates.sort(key=lambda x: x[1], reverse=True)
        slots = self.config.max_positions - len(self.open_positions)

        for symbol, gap_ratio, price, iv in candidates[:slots]:
            self._enter_iron_condor(current_date, symbol, gap_ratio, price, iv)

    def _enter_iron_condor(
        self,
        entry_date: Any,
        symbol: str,
        gap_ratio: float,
        price: float,
        iv: float,
    ) -> None:
        """Size and open a new iron condor, appending to open_positions."""
        monthly_move = price * iv * math.sqrt(self.config.target_dte / 365.0)
        wing = self.config.wing_width
        # Credit: 40% of wing, capped at 30% of expected monthly move
        credit = min(wing * 0.40, monthly_move * 0.30)
        max_risk = wing - credit
        # Size by risk budget, capped at max_position_pct of capital
        contracts = max(1, int(self.capital * self.config.max_position_pct / (max_risk * 100.0)))
        # Four-legged cost (sell/buy call + put wings)
        entry_cost = (
            self.config.commission_per_contract * 4 * contracts
            + self.config.slippage_per_leg * 4 * contracts
        )

        trade = IronCondorTrade(
            symbol=symbol,
            entry_date=str(entry_date),
            entry_credit=credit * contracts * 100.0 - entry_cost,
            max_risk=max_risk * contracts * 100.0,
            gap_ratio=gap_ratio,
            dte_at_entry=self.config.target_dte,
        )
        trade._entry_price = price  # for delta P&L estimation
        self.open_positions.append(trade)

        logger.debug(
            "Opened IC %s gap=%.2f credit=$%.2f max_risk=$%.2f contracts=%d",
            symbol,
            gap_ratio,
            trade.entry_credit,
            trade.max_risk,
            contracts,
        )

    def _check_exits(
        self,
        current_date: Any,
        price_data: pd.DataFrame,
    ) -> None:
        """Evaluate exit conditions: profit target, stop loss, or time exit."""
        for pos in list(self.open_positions):
            entry_ts = pd.Timestamp(pos.entry_date)
            current_ts = pd.Timestamp(str(current_date))
            days_held = (current_ts - entry_ts).days

            # Verify underlying still has data on this date
            sym_data = price_data[
                (price_data["symbol"] == pos.symbol)
                & (price_data["date"] == current_date)
            ]
            if sym_data.empty:
                continue

            mtm = self._mark_to_market(pos, current_date, price_data)
            if mtm >= pos.entry_credit * self.config.profit_target_pct:
                self._close_position(pos, current_date, "profit_target", mtm)
            elif mtm <= -(pos.max_risk * 0.50):
                self._close_position(pos, current_date, "stop_loss", mtm)
            elif days_held >= (self.config.target_dte - self.config.exit_dte):
                self._close_position(pos, current_date, "time_exit", mtm)

    def _close_position(
        self,
        pos: IronCondorTrade,
        exit_date: Any,
        reason: str,
        mtm: float,
    ) -> None:
        """Realise P&L, update capital, and move position to closed log."""
        exit_cost = (
            self.config.commission_per_contract * 4
            + self.config.slippage_per_leg * 4
        )

        pos.exit_date = str(exit_date)
        pos.exit_reason = reason
        pos.pnl = pos.entry_credit + mtm - exit_cost
        pos.pnl_pct = pos.pnl / pos.max_risk if pos.max_risk > 0.0 else 0.0
        pos.is_winner = pos.pnl > 0.0

        self.capital += pos.pnl
        self.trades.append(pos)
        self.open_positions.remove(pos)

        logger.debug(
            "Closed IC %s reason=%s pnl=$%.2f capital=$%.2f",
            pos.symbol,
            reason,
            pos.pnl,
            self.capital,
        )

    def _load_vix(self) -> None:
        """Load VIX history from CBOE CSV cache."""
        vix_path = Path("data/cache/vix_history.csv")
        if vix_path.exists():
            df = pd.read_csv(vix_path)
            self._vix_data = {}
            for _, row in df.iterrows():
                try:
                    d = pd.Timestamp(row["DATE"]).date()
                    self._vix_data[d] = float(row["CLOSE"])
                except Exception:
                    pass
            logger.info("Loaded %d VIX observations", len(self._vix_data))
        else:
            logger.warning("No VIX history found at %s — using flat proxy", vix_path)
            self._vix_data = {}

    def _get_vix(self, current_date: Any) -> float:
        """Get VIX close for a date, with fallback."""
        d = current_date if isinstance(current_date, date) else pd.Timestamp(str(current_date)).date()
        # Try exact date, then look back up to 5 days
        for offset in range(6):
            lookup = d - timedelta(days=offset)
            if lookup in self._vix_data:
                return self._vix_data[lookup]
        return 20.0  # long-term VIX average fallback

    def _mark_to_market(
        self,
        pos: IronCondorTrade,
        current_date: Any,
        price_data: pd.DataFrame | None = None,
    ) -> float:
        """Estimate P&L using theta decay minus adverse underlying move impact.

        Models:
        - Theta: linear decay of premium over holding period
        - Delta: loss from underlying moving against the short strikes
        - This is approximate but produces realistic win/loss distributions
        """
        days_held = max(
            1,
            (pd.Timestamp(str(current_date)) - pd.Timestamp(pos.entry_date)).days,
        )
        theta_fraction = days_held / max(pos.dte_at_entry, 1)

        # Theta gain (premium decays in our favor)
        theta_pnl = pos.entry_credit * theta_fraction * 0.6

        # Delta loss (underlying move impact)
        delta_loss = 0.0
        if price_data is not None and hasattr(pos, '_entry_price'):
            sym_now = price_data[
                (price_data["symbol"] == pos.symbol) &
                (price_data["date"] == current_date)
            ]
            if not sym_now.empty:
                current_price = float(sym_now["close"].values[0])
                move_pct = abs(current_price - pos._entry_price) / pos._entry_price
                # Iron condor loses when underlying moves beyond short strikes
                # Approximate: loss scales quadratically with move size
                if move_pct > 0.03:  # beyond ~1 std dev move
                    delta_loss = pos.max_risk * min(1.0, ((move_pct - 0.03) / 0.07) ** 1.5)

        return theta_pnl - delta_loss

    @staticmethod
    def _realized_vol(closes: np.ndarray, window: int) -> float:
        """Annualised close-to-close realised volatility. Returns 0.0 on insufficient data."""
        if len(closes) < window + 1:
            return 0.0
        recent = closes[-(window + 1):]
        log_returns = np.diff(np.log(recent))
        if len(log_returns) < 2:
            return 0.0
        return float(np.std(log_returns, ddof=1) * np.sqrt(252))

    def _compute_results(self) -> dict[str, Any]:
        """Aggregate trades and equity curve; delegate to metrics.compute_all."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent))
        from metrics import compute_all  # noqa: PLC0415

        if not self.trades:
            return {
                "trades": [],
                "equity_curve": self.equity_curve,
                "metrics": {},
            }

        equity_series = pd.Series(
            [e["equity"] for e in self.equity_curve],
            name="equity",
            dtype=float,
        )
        returns = equity_series.pct_change().dropna()

        trades_df = pd.DataFrame(
            [
                {
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "entry_date": t.entry_date,
                    "exit_date": t.exit_date,
                    "symbol": t.symbol,
                    "gap_ratio": t.gap_ratio,
                    "exit_reason": t.exit_reason,
                    "is_winner": t.is_winner,
                }
                for t in self.trades
            ]
        )

        metrics = compute_all(trades_df, equity_series, returns)
        metrics["initial_capital"] = self.config.initial_capital
        metrics["final_capital"] = float(self.capital)
        metrics["total_return_pct"] = (
            (self.capital - self.config.initial_capital) / self.config.initial_capital * 100.0
        )
        wins = [t for t in self.trades if t.is_winner]
        losses = [t for t in self.trades if not t.is_winner]
        metrics["avg_gap_ratio_winners"] = float(np.mean([t.gap_ratio for t in wins])) if wins else 0.0
        metrics["avg_gap_ratio_losers"] = float(np.mean([t.gap_ratio for t in losses])) if losses else 0.0
        exit_reasons: dict[str, int] = {}
        for t in self.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        metrics["exit_reasons"] = exit_reasons

        return {
            "trades": trades_df.to_dict("records"),
            "equity_curve": self.equity_curve,
            "metrics": metrics,
        }


def generate_synthetic_data(
    symbols: list[str] | None = None,
    start: str = "2015-01-01",
    end: str = "2025-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate GBM synthetic price data for backtester mechanics testing.

    NOTE: Results have zero information about real strategy edge. Use
    ``backtest.data_provider.PolygonDataProvider`` for strategy research.

    Args:
        symbols: Tickers to generate; defaults to 10 large-cap names.
        start: Series start date (YYYY-MM-DD).
        end: Series end date (YYYY-MM-DD).
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame with columns [date, symbol, close, volume].
    """
    if symbols is None:
        symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "NVDA", "TSLA", "JPM", "V", "JNJ",
        ]

    np.random.seed(seed)
    dates = pd.bdate_range(start, end)
    rows: list[dict[str, Any]] = []

    for sym in symbols:
        price = np.random.uniform(50.0, 300.0)
        vol = np.random.uniform(0.15, 0.45)

        for d in dates:
            daily_return = np.random.normal(0.0003, vol / np.sqrt(252))
            price = price * (1.0 + daily_return)
            volume = int(np.random.lognormal(15.0, 1.0))
            rows.append(
                {
                    "date": d.date(),
                    "symbol": sym,
                    "close": round(price, 4),
                    "volume": volume,
                }
            )

    return pd.DataFrame(rows)


def _print_results(metrics: dict[str, Any]) -> None:
    """Pretty-print backtest metrics to stdout."""
    print("\n" + "=" * 62)
    print("  HV-IV GAP BACKTEST RESULTS")
    print("=" * 62)
    ordered_keys = [
        "total_trades", "win_rate", "profit_factor", "total_return_pct",
        "sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown",
        "total_pnl", "avg_pnl_per_trade", "initial_capital", "final_capital",
        "avg_gap_ratio_winners", "avg_gap_ratio_losers",
    ]
    for key in ordered_keys:
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                print(f"  {key:<30} {val:>12.4f}")
            else:
                print(f"  {key:<30} {val!s:>12}")
    if "exit_reasons" in metrics:
        print(f"\n  Exit reasons: {metrics['exit_reasons']}")
    print("=" * 62)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="HV-IV Gap Strategy Backtest")
    parser.add_argument("--start", default="2015-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2025-01-01", help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=25_000.0, help="Initial capital ($)")
    parser.add_argument("--gap-ratio", type=float, default=1.2, help="Min IV/HV gap ratio")
    parser.add_argument("--max-pos", type=int, default=5, help="Max simultaneous positions")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic GBM data for mechanics testing (not strategy validation)",
    )
    args = parser.parse_args()

    config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        min_gap_ratio=args.gap_ratio,
        max_positions=args.max_pos,
    )

    bt = HVIVBacktester(config)

    if args.synthetic:
        logger.warning(
            "SYNTHETIC DATA MODE: results indicate mechanics only, "
            "not real strategy performance."
        )
        data = generate_synthetic_data(start=args.start, end=args.end)
    else:
        # Load real data from Polygon cache
        # Prefer the full multi-year file, fall back to truncated
        cache_file = Path("data/cache/sp500_top30_full.json")
        if not cache_file.exists():
            cache_file = Path("data/cache/sp500_top30.json")
        if not cache_file.exists():
            print("ERROR: No cached data found. Run the data fetcher first:")
            print("  python -m backtest.hv_iv_backtest --synthetic  (for testing)")
            print("  Or fetch real data via backtest/data_provider.py")
            sys.exit(1)

        import json

        logger.info("Loading real Polygon data from %s", cache_file)
        with open(cache_file) as f:
            raw = json.load(f)

        rows = []
        for symbol, bars in raw.items():
            for bar in bars:
                bar_date = date.fromtimestamp(bar["t"] / 1000)
                rows.append({
                    "date": bar_date,
                    "symbol": symbol,
                    "close": bar["c"],
                    "volume": bar.get("v", 0),
                })

        data = pd.DataFrame(rows)
        logger.info(
            "Loaded %d bars for %d symbols (%s to %s)",
            len(data), data["symbol"].nunique(),
            data["date"].min(), data["date"].max(),
        )

    results = bt.run(data)
    _print_results(results["metrics"])

    total = results["metrics"].get("total_trades", 0)
    if total == 0:
        print("\nNo trades generated — try lowering --gap-ratio or widening the date range.")
    else:
        # Run verification
        equity = [e["equity"] for e in results["equity_curve"]]
        if len(equity) > 1:
            daily_returns = np.diff(equity) / np.array(equity[:-1])

            from backtest.verification import StrategyVerifier

            verifier = StrategyVerifier(num_strategies_tested=1)
            vresult = verifier.verify(
                daily_returns,
                trades=results["trades"],
                strategy_name="HV-IV Gap (real data)",
            )
            print(vresult.summary())

        # Show sample trades
        if results["trades"]:
            print("\nSample of last 5 trades:")
            trades_df = pd.DataFrame(results["trades"])
            print(trades_df.tail().to_string(index=False))
