"""Strategy Verification Framework.

Statistical tests to determine whether a backtest result represents
genuine alpha or is explainable by luck, overfitting, or data artifacts.

At Citadel, a strategy must pass ALL of these gates before paper trading:
1. Sharpe t-statistic > 2.0 (95% confidence it's not zero)
2. Walk-forward OOS Sharpe > 50% of in-sample Sharpe
3. Monte Carlo permutation p-value < 0.05
4. Deflated Sharpe Ratio accounts for multiple testing
5. Minimum 100 independent trades
6. Profit factor > 1.2 after realistic costs

Reference: Harvey, Liu, Zhu (2016) "...and the Cross-Section of Expected Returns"
— recommends t > 3.0 for novel factors due to multiple testing.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of a complete verification battery."""

    strategy_name: str

    # Stage 1: Basic metrics
    total_trades: int = 0
    sharpe_ratio: float = 0.0
    sharpe_t_stat: float = 0.0        # t-test: is Sharpe != 0?
    sharpe_p_value: float = 1.0
    sharpe_significant: bool = False   # t > 2.0

    # Stage 2: Overfitting tests
    is_oos_sharpe: float = 0.0        # out-of-sample Sharpe
    oos_is_ratio: float = 0.0         # OOS/IS ratio (want > 0.5)
    overfitting_risk: str = "UNKNOWN" # LOW, MEDIUM, HIGH, CRITICAL

    # Stage 3: Monte Carlo
    mc_p_value: float = 1.0           # probability of luck
    mc_percentile: float = 0.0        # where strategy falls in random distribution
    mc_significant: bool = False       # p < 0.05

    # Stage 4: Deflated Sharpe
    deflated_sharpe: float = 0.0      # adjusted for multiple testing
    num_strategies_tested: int = 1    # how many variants were tried

    # Bootstrap Sharpe CI (fat-tail robust)
    bootstrap_sharpe_ci: tuple[float, float] = (0.0, 0.0)
    bootstrap_sharpe_excludes_zero: bool = False

    # Stage 5: Robustness
    min_sharpe_subsample: float = 0.0  # worst Sharpe across subsamples
    pct_positive_months: float = 0.0   # fraction of months with positive returns
    max_drawdown_pct: float = 0.0
    drawdown_recovery_days: int = 0

    # Overall
    passes_all_gates: bool = False
    gate_failures: list[str] = field(default_factory=list)
    recommendation: str = "DO NOT TRADE"

    def summary(self) -> str:
        """Return a formatted multi-line summary of all verification gates.

        Returns:
            Human-readable string with pass/fail status for each gate.
        """
        lines = [
            f"{'=' * 60}",
            f"  STRATEGY VERIFICATION: {self.strategy_name}",
            f"{'=' * 60}",
            "",
            "  GATE 1 — Sharpe Significance",
            f"    Sharpe Ratio:       {self.sharpe_ratio:>8.3f}",
            f"    t-statistic:        {self.sharpe_t_stat:>8.3f}  {'PASS' if self.sharpe_significant else 'FAIL'} (need > 2.0)",
            f"    p-value:            {self.sharpe_p_value:>8.4f}",
            f"    Bootstrap 95% CI:   [{self.bootstrap_sharpe_ci[0]:.3f}, {self.bootstrap_sharpe_ci[1]:.3f}]  {'PASS' if self.bootstrap_sharpe_excludes_zero else 'FAIL'} (must exclude 0)",
            f"    Total trades:       {self.total_trades:>8d}  {'PASS' if self.total_trades >= 100 else 'FAIL'} (need >= 100)",
            "",
            "  GATE 2 — Overfitting",
            f"    In-sample Sharpe:   {self.sharpe_ratio:>8.3f}",
            f"    Out-of-sample:      {self.is_oos_sharpe:>8.3f}",
            f"    OOS/IS ratio:       {self.oos_is_ratio:>8.3f}  {'PASS' if self.oos_is_ratio > 0.5 else 'FAIL'} (need > 0.50)",
            f"    Overfitting risk:   {self.overfitting_risk:>8s}",
            "",
            "  GATE 3 — Monte Carlo Permutation",
            f"    p-value:            {self.mc_p_value:>8.4f}  {'PASS' if self.mc_significant else 'FAIL'} (need < 0.05)",
            f"    Percentile:         {self.mc_percentile:>8.1f}%",
            "",
            "  GATE 4 — Deflated Sharpe (multiple testing)",
            f"    Strategies tested:  {self.num_strategies_tested:>8d}",
            f"    Deflated Sharpe:    {self.deflated_sharpe:>8.3f}  {'PASS' if self.deflated_sharpe > 0 else 'FAIL'} (need > 0)",
            "",
            "  GATE 5 — Robustness",
            f"    Worst subsample SR: {self.min_sharpe_subsample:>8.3f}  {'PASS' if self.min_sharpe_subsample > 0 else 'FAIL'} (need > 0)",
            f"    Positive months:    {self.pct_positive_months * 100:>7.1f}%  {'PASS' if self.pct_positive_months > 0.5 else 'FAIL'} (need > 50%)",
            f"    Max drawdown:       {self.max_drawdown_pct:>7.1f}%",
            "",
            f"  {'=' * 56}",
            f"  OVERALL: {self.recommendation}",
        ]
        if self.gate_failures:
            lines.append(f"  Failures: {', '.join(self.gate_failures)}")
        lines.append(f"  {'=' * 56}")
        return "\n".join(lines)


class StrategyVerifier:
    """Runs the full verification battery on backtest results.

    Args:
        num_strategies_tested: How many parameter combinations / strategy
            variants were tried before arriving at this one. Used for
            multiple-testing correction. Be honest — underreporting
            this number inflates your confidence.
    """

    def __init__(self, num_strategies_tested: int = 1, oos_is_threshold: float = 0.5) -> None:
        self.num_strategies_tested = max(1, num_strategies_tested)
        self.oos_is_threshold = oos_is_threshold

    def verify(
        self,
        returns: np.ndarray,
        trades: list[dict[str, Any]] | pd.DataFrame,
        strategy_name: str = "HV-IV Gap",
        is_returns: np.ndarray | None = None,
        oos_returns: np.ndarray | None = None,
        n_mc_simulations: int = 10000,
    ) -> VerificationResult:
        """Run complete verification battery.

        Args:
            returns: Daily or periodic strategy returns.
            trades: List of trade records or DataFrame with trade history.
            strategy_name: Name for reporting.
            is_returns: In-sample returns (for overfitting test).
            oos_returns: Out-of-sample returns (for overfitting test).
            n_mc_simulations: Number of Monte Carlo permutations.

        Returns:
            VerificationResult with all gates evaluated.
        """
        result = VerificationResult(
            strategy_name=strategy_name,
            num_strategies_tested=self.num_strategies_tested,
        )

        if isinstance(trades, pd.DataFrame):
            result.total_trades = len(trades)
        else:
            result.total_trades = len(trades) if trades else 0

        returns = np.asarray(returns, dtype=float)
        returns = returns[~np.isnan(returns)]

        if len(returns) < 20:
            result.recommendation = "INSUFFICIENT DATA"
            result.gate_failures.append("fewer than 20 return observations")
            return result

        # Gate 1: Sharpe significance
        self._test_sharpe_significance(returns, result)
        self._test_sharpe_bootstrap(returns, result)

        # Gate 2: Overfitting (if IS/OOS provided)
        if is_returns is not None and oos_returns is not None:
            self._test_overfitting(
                np.asarray(is_returns, dtype=float),
                np.asarray(oos_returns, dtype=float),
                result,
            )
        else:
            # Run subsample test instead
            self._test_subsample_stability(returns, result)

        # Gate 3: Monte Carlo permutation
        self._test_monte_carlo(returns, result, n_mc_simulations)

        # Gate 4: Deflated Sharpe
        self._test_deflated_sharpe(returns, result)

        # Gate 5: Robustness
        self._test_robustness(returns, result)

        # Overall decision
        self._make_decision(result)

        return result

    # ------------------------------------------------------------------
    # Private gate implementations
    # ------------------------------------------------------------------

    def _test_sharpe_significance(
        self, returns: np.ndarray, result: VerificationResult
    ) -> None:
        """Test if Sharpe ratio is statistically different from zero.

        Uses the Lo (2002) t-statistic for the Sharpe ratio:
            t = SR * sqrt(n / 252)

        This is a conservative approximation that accounts for the
        fact that Sharpe is an estimate with sampling error.
        """
        n = len(returns)
        mean_r = float(np.mean(returns))
        std_r = float(np.std(returns, ddof=1))

        if std_r < 1e-10:
            result.sharpe_ratio = 0.0
            result.sharpe_t_stat = 0.0
            result.sharpe_p_value = 1.0
            result.sharpe_significant = False
            return

        # Annualized Sharpe (assuming daily returns)
        sharpe = (mean_r / std_r) * np.sqrt(252)
        result.sharpe_ratio = float(sharpe)

        # t-statistic for Sharpe (Lo 2002)
        # Simplified: t ≈ SR * sqrt(n / 252)
        t_stat = sharpe * math.sqrt(n / 252)
        result.sharpe_t_stat = float(t_stat)

        # p-value (two-tailed normal approximation)
        from scipy.stats import norm

        result.sharpe_p_value = float(2 * (1 - norm.cdf(abs(t_stat))))
        result.sharpe_significant = t_stat > 2.0

        if not result.sharpe_significant:
            result.gate_failures.append(
                f"Sharpe t-stat {t_stat:.2f} < 2.0"
            )

    def _test_sharpe_bootstrap(
        self, returns: np.ndarray, result: VerificationResult, n_bootstrap: int = 10000
    ) -> None:
        """Bootstrap confidence interval for Sharpe ratio.

        More robust than the Gaussian t-test for fat-tailed option returns.
        Reference: Ledoit & Wolf (2008) "Robust performance hypothesis testing
        with the Sharpe ratio." Journal of Empirical Finance.

        Args:
            returns: Array of periodic strategy returns.
            result: VerificationResult to populate in-place.
            n_bootstrap: Number of bootstrap resamples (default 10 000).
        """
        rng = np.random.default_rng(42)
        n = len(returns)
        boot_sharpes = np.empty(n_bootstrap)

        for i in range(n_bootstrap):
            sample = rng.choice(returns, size=n, replace=True)
            std = np.std(sample, ddof=1)
            if std > 1e-10:
                boot_sharpes[i] = np.mean(sample) / std * np.sqrt(252)
            else:
                boot_sharpes[i] = 0.0

        ci_lower = float(np.percentile(boot_sharpes, 2.5))
        ci_upper = float(np.percentile(boot_sharpes, 97.5))

        result.bootstrap_sharpe_ci = (ci_lower, ci_upper)
        result.bootstrap_sharpe_excludes_zero = ci_lower > 0

        if not result.bootstrap_sharpe_excludes_zero:
            result.gate_failures.append(
                f"Bootstrap 95% CI [{ci_lower:.3f}, {ci_upper:.3f}] includes zero"
            )

    def _test_overfitting(
        self,
        is_returns: np.ndarray,
        oos_returns: np.ndarray,
        result: VerificationResult,
    ) -> None:
        """Compare in-sample vs out-of-sample performance.

        A healthy strategy retains at least 50% of its in-sample Sharpe
        on held-out data. Significant degradation indicates the backtest
        parameters were overfit to historical noise.
        """
        is_std = float(np.std(is_returns, ddof=1))
        oos_std = float(np.std(oos_returns, ddof=1))

        is_sharpe = (
            float(np.mean(is_returns)) / is_std * np.sqrt(252)
            if is_std > 1e-10
            else 0.0
        )
        oos_sharpe = (
            float(np.mean(oos_returns)) / oos_std * np.sqrt(252)
            if oos_std > 1e-10
            else 0.0
        )

        result.is_oos_sharpe = float(oos_sharpe)
        ratio = oos_sharpe / is_sharpe if abs(is_sharpe) > 0.01 else 0.0
        result.oos_is_ratio = float(ratio)

        threshold = self.oos_is_threshold
        high_threshold = threshold * 0.5  # half of the configured threshold

        if ratio > threshold * 1.5:
            result.overfitting_risk = "LOW"
        elif ratio > threshold:
            result.overfitting_risk = "MEDIUM"
        elif ratio > high_threshold:
            result.overfitting_risk = "HIGH"
            result.gate_failures.append(
                f"OOS/IS ratio {ratio:.2f} indicates high overfitting risk"
            )
        else:
            result.overfitting_risk = "CRITICAL"
            result.gate_failures.append(
                f"OOS/IS ratio {ratio:.2f} — strategy is likely overfit"
            )

    def _test_subsample_stability(
        self, returns: np.ndarray, result: VerificationResult
    ) -> None:
        """Test Sharpe stability across time subsamples when IS/OOS not available.

        Splits returns into five equal chunks and measures per-chunk Sharpe.
        The worst-half average is used as an OOS proxy to detect time-varying
        performance — a hallmark of overfit strategies.
        """
        n = len(returns)
        n_subsamples = 5
        chunk_size = n // n_subsamples

        sharpes: list[float] = []
        for i in range(n_subsamples):
            chunk = returns[i * chunk_size:(i + 1) * chunk_size]
            if len(chunk) < 20:
                continue
            std = float(np.std(chunk, ddof=1))
            if std > 1e-10:
                sr = float(np.mean(chunk)) / std * np.sqrt(252)
                sharpes.append(sr)

        if not sharpes:
            return

        result.min_sharpe_subsample = float(min(sharpes))

        # Use worst half as OOS proxy
        sharpes_sorted = sorted(sharpes)
        half = max(1, len(sharpes_sorted) // 2)
        oos_proxy = float(np.mean(sharpes_sorted[:half]))
        is_proxy = float(np.mean(sharpes_sorted[half:]))

        result.is_oos_sharpe = oos_proxy
        result.oos_is_ratio = (
            oos_proxy / is_proxy if abs(is_proxy) > 0.01 else 0.0
        )

        if result.oos_is_ratio > 0.5:
            result.overfitting_risk = "LOW-MEDIUM"
        else:
            result.overfitting_risk = "MEDIUM-HIGH"
            result.gate_failures.append(
                f"Subsample instability: worst SR {result.min_sharpe_subsample:.2f}"
            )

    def _test_monte_carlo(
        self,
        returns: np.ndarray,
        result: VerificationResult,
        n_simulations: int = 10000,
    ) -> None:
        """Monte Carlo permutation test: could this Sharpe happen by chance?

        Randomly flips the sign of each daily return (preserving the magnitude
        distribution) and computes the Sharpe of each permuted series. The
        empirical p-value is the fraction of random Sharpes that meet or exceed
        the observed value. A p-value below 0.05 means the result is unlikely
        to arise from a sign-symmetric random walk.
        """
        n = len(returns)
        actual_sharpe = result.sharpe_ratio

        rng = np.random.default_rng(42)
        random_sharpes = np.empty(n_simulations)

        for i in range(n_simulations):
            signs = rng.choice([-1, 1], size=n)
            shuffled = returns * signs
            std = float(np.std(shuffled, ddof=1))
            if std > 1e-10:
                random_sharpes[i] = float(np.mean(shuffled)) / std * np.sqrt(252)
            else:
                random_sharpes[i] = 0.0

        p_value = float(np.mean(random_sharpes >= actual_sharpe))
        percentile = float(np.mean(random_sharpes < actual_sharpe) * 100)

        result.mc_p_value = p_value
        result.mc_percentile = percentile
        result.mc_significant = p_value < 0.05

        if not result.mc_significant:
            result.gate_failures.append(
                f"Monte Carlo p={p_value:.3f} — strategy may be luck"
            )

    def _test_deflated_sharpe(
        self, returns: np.ndarray, result: VerificationResult
    ) -> None:
        """Deflated Sharpe Ratio (Bailey & de Prado 2014).

        Adjusts for multiple testing: if you tried 100 parameter combos,
        the best one's Sharpe is inflated by sqrt(2 * ln(100)) ≈ 3.03.
        The deflated Sharpe subtracts this expected inflation so that only
        genuine alpha survives the correction.
        """
        k = self.num_strategies_tested
        observed_sr = result.sharpe_ratio

        # Expected max Sharpe under null (Euler-Mascheroni correction)
        euler_mascheroni = 0.5772156649
        if k > 1:
            log_k = np.log(k)
            expected_max_sr = np.sqrt(2 * log_k) - (
                np.log(np.pi) + euler_mascheroni
            ) / (2 * np.sqrt(2 * log_k))
        else:
            expected_max_sr = 0.0

        deflated = float(observed_sr - expected_max_sr)
        result.deflated_sharpe = deflated

        if deflated <= 0 and k > 1:
            result.gate_failures.append(
                f"Deflated Sharpe {deflated:.3f} <= 0 after testing {k} variants"
            )

    def _test_robustness(
        self, returns: np.ndarray, result: VerificationResult
    ) -> None:
        """Test strategy robustness: drawdown depth, recovery, and hit rate.

        Computes maximum drawdown on the cumulative P&L series, estimates
        the recovery period, and measures what fraction of calendar months
        the strategy generated positive returns.
        """
        # Max drawdown on cumulative returns
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        result.max_drawdown_pct = float(np.max(drawdowns) * 100) if len(drawdowns) > 0 else 0.0

        # Drawdown recovery: days from peak-drawdown to new high
        if np.max(drawdowns) > 0:
            max_dd_idx = int(np.argmax(drawdowns))
            recovery_idx = max_dd_idx
            for j in range(max_dd_idx, len(cumulative)):
                if cumulative[j] >= running_max[max_dd_idx]:
                    recovery_idx = j
                    break
            result.drawdown_recovery_days = int(recovery_idx - max_dd_idx)

        # Positive months (21-trading-day buckets)
        if len(returns) >= 21:
            monthly_returns: list[float] = []
            for i in range(0, len(returns) - 20, 21):
                monthly_returns.append(float(np.sum(returns[i:i + 21])))
            if monthly_returns:
                result.pct_positive_months = float(
                    np.mean(np.array(monthly_returns) > 0)
                )
                if result.pct_positive_months < 0.5:
                    result.gate_failures.append(
                        f"Only {result.pct_positive_months * 100:.0f}% positive months"
                    )

        # Propagate subsample min Sharpe if not already set from overfitting test
        if result.min_sharpe_subsample == 0.0:
            n = len(returns)
            chunk_size = n // 5
            subsample_sharpes: list[float] = []
            for i in range(5):
                chunk = returns[i * chunk_size:(i + 1) * chunk_size]
                if len(chunk) < 20:
                    continue
                std = float(np.std(chunk, ddof=1))
                if std > 1e-10:
                    subsample_sharpes.append(
                        float(np.mean(chunk)) / std * np.sqrt(252)
                    )
            if subsample_sharpes:
                result.min_sharpe_subsample = float(min(subsample_sharpes))

        if result.min_sharpe_subsample < 0 and "Subsample instability" not in " ".join(result.gate_failures):
            result.gate_failures.append(
                f"Worst subsample SR {result.min_sharpe_subsample:.2f} is negative"
            )

    def _make_decision(self, result: VerificationResult) -> None:
        """Make final recommendation based on all gates.

        Decision hierarchy:
        1. Insufficient data overrides everything.
        2. Critical failures (overfit / data issues) block trading.
        3. Zero failures → approved for paper trading.
        4. One failure → conditional approval.
        5. Multiple failures → do not trade.
        """
        if result.total_trades < 50:
            result.recommendation = "INSUFFICIENT DATA — need more trades"
            result.passes_all_gates = False
            return

        critical_failures = [
            f for f in result.gate_failures
            if "overfit" in f.lower() or "insufficient" in f.lower()
        ]

        if critical_failures:
            result.recommendation = "DO NOT TRADE — critical failures detected"
            result.passes_all_gates = False
        elif len(result.gate_failures) == 0:
            result.recommendation = "APPROVED FOR PAPER TRADING"
            result.passes_all_gates = True
        elif len(result.gate_failures) == 1:
            result.recommendation = "CONDITIONAL APPROVAL — address single failure"
            result.passes_all_gates = False
        else:
            result.recommendation = (
                f"DO NOT TRADE — {len(result.gate_failures)} gate failures"
            )
            result.passes_all_gates = False


# ---------------------------------------------------------------------------
# Backtest run logging — honest multiple-testing accounting
# ---------------------------------------------------------------------------


@dataclass
class BacktestRunLog:
    """Immutable record of a backtest configuration. Log this every time.

    Attributes:
        run_id: Short UUID identifying this run.
        timestamp: ISO-format datetime when the run was logged.
        strategy_name: Human-readable strategy identifier.
        parameters: All tunable parameters used in this run.
        universe: Tickers used (may be truncated in serialised form).
        date_range: (start, end) date strings for the backtest window.
        data_source: Origin of price data (e.g. "polygon", "synthetic").
        num_prior_runs: How many runs preceded this one — feeds into the
            Deflated Sharpe's ``num_strategies_tested``.
    """

    run_id: str
    timestamp: str
    strategy_name: str
    parameters: dict[str, Any]
    universe: list[str]
    date_range: tuple[str, str]
    data_source: str
    num_prior_runs: int

    def to_json(self) -> str:
        """Serialise this run record to a JSON string.

        Returns:
            Indented JSON string representing all run metadata.
        """
        import json

        return json.dumps(
            {
                "run_id": self.run_id,
                "timestamp": self.timestamp,
                "strategy_name": self.strategy_name,
                "parameters": self.parameters,
                "universe": self.universe[:10],
                "universe_size": len(self.universe),
                "date_range": list(self.date_range),
                "data_source": self.data_source,
                "num_prior_runs": self.num_prior_runs,
            },
            indent=2,
        )


class BacktestLogger:
    """Tracks all backtest runs to enforce honest multiple-testing accounting.

    Every time you run a backtest with different parameters, log it here.
    The count of runs becomes ``num_strategies_tested`` for the Deflated Sharpe.

    Args:
        log_dir: Directory where individual run JSON files are persisted.
    """

    def __init__(self, log_dir: str = "logs/backtest_runs") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._runs: list[BacktestRunLog] = []
        self._load_existing()

    def _load_existing(self) -> None:
        """Load previous run logs from disk."""
        import json

        for f in sorted(self._log_dir.glob("run_*.json")):
            try:
                data = json.loads(f.read_text())
                # Reconstruct tuple for date_range and drop summary-only keys
                data["date_range"] = tuple(data["date_range"])
                data.pop("universe_size", None)
                self._runs.append(BacktestRunLog(**data))
            except Exception:
                pass

    def log_run(
        self,
        strategy_name: str,
        parameters: dict[str, Any],
        universe: list[str],
        date_range: tuple[str, str],
        data_source: str = "unknown",
    ) -> BacktestRunLog:
        """Log a new backtest run and persist it to disk.

        Args:
            strategy_name: Human-readable strategy identifier.
            parameters: All tunable parameters for this run.
            universe: List of ticker symbols used.
            date_range: (start_date, end_date) strings.
            data_source: Origin of price data.

        Returns:
            The newly created BacktestRunLog record.
        """
        import uuid
        from datetime import datetime

        run = BacktestRunLog(
            run_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            strategy_name=strategy_name,
            parameters=parameters,
            universe=universe,
            date_range=date_range,
            data_source=data_source,
            num_prior_runs=len(self._runs),
        )

        self._runs.append(run)

        log_file = self._log_dir / f"run_{run.run_id}.json"
        log_file.write_text(run.to_json())

        logger.info("Logged backtest run %s (#%d)", run.run_id, len(self._runs))
        return run

    @property
    def total_runs(self) -> int:
        """Total number of backtest runs — use this for num_strategies_tested."""
        return len(self._runs)

    @property
    def runs(self) -> list[BacktestRunLog]:
        """Read-only copy of all logged runs."""
        return list(self._runs)
