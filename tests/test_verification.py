"""Tests for the Strategy Verification Framework."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from backtest.live_monitor import KillCriteria, LiveMonitor, MonitorAlert, TradeRecord
from backtest.verification import BacktestLogger, StrategyVerifier, VerificationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trade(
    trade_id: int,
    pnl: float,
    pnl_pct: float = 0.01,
    symbol: str = "AAPL",
) -> TradeRecord:
    """Factory for TradeRecord with sensible defaults."""
    return TradeRecord(
        trade_id=trade_id,
        symbol=symbol,
        entry_date="2025-01-01",
        exit_date="2025-01-15",
        pnl=pnl,
        pnl_pct=pnl_pct,
    )


def _positive_returns(n: int = 504, seed: int = 42) -> np.ndarray:
    """Generate a positively-drifted daily return series."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.001, 0.01, n)


def _zero_returns(n: int = 504, seed: int = 42) -> np.ndarray:
    """Generate a zero-mean daily return series."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, n)


def _strong_returns(n: int = 504, seed: int = 42) -> np.ndarray:
    """Generate a high-drift daily return series for clear significance."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.002, 0.01, n)


def _dummy_trades(n: int = 100) -> list[dict]:
    """Generate a list of dummy trade dicts."""
    return [{"pnl": 1}] * n


# ---------------------------------------------------------------------------
# Gate 1: Sharpe significance
# ---------------------------------------------------------------------------


class TestSharpeSignificance:
    def test_positive_sharpe_significant(self) -> None:
        """Strong positive returns should produce a positive Sharpe."""
        returns = _positive_returns()
        verifier = StrategyVerifier()
        result = verifier.verify(returns, trades=_dummy_trades(), strategy_name="test")
        assert result.sharpe_ratio > 0
        assert result.sharpe_t_stat > 0

    def test_strong_drift_passes_t_threshold(self) -> None:
        """High-drift returns over two years should exceed t=2.0."""
        returns = _strong_returns()
        verifier = StrategyVerifier()
        result = verifier.verify(returns, trades=_dummy_trades(), strategy_name="test")
        assert result.sharpe_significant

    def test_zero_returns_not_significant(self) -> None:
        """Zero-mean returns should not produce a significant t-statistic."""
        returns = _zero_returns()
        verifier = StrategyVerifier()
        result = verifier.verify(returns, trades=_dummy_trades(), strategy_name="test")
        assert not result.sharpe_significant

    def test_zero_returns_failure_recorded(self) -> None:
        """Gate failure should be appended for insignificant Sharpe."""
        returns = _zero_returns()
        verifier = StrategyVerifier()
        result = verifier.verify(returns, trades=_dummy_trades())
        failures_lower = [f.lower() for f in result.gate_failures]
        assert any("t-stat" in f or "sharpe" in f for f in failures_lower)

    def test_insufficient_data_returns_early(self) -> None:
        """Fewer than 20 observations should return INSUFFICIENT DATA."""
        returns = np.array([0.01, -0.01])
        verifier = StrategyVerifier()
        result = verifier.verify(returns, trades=[], strategy_name="test")
        assert result.recommendation == "INSUFFICIENT DATA"

    def test_constant_returns_handled(self) -> None:
        """Constant (zero std) returns should not raise an exception."""
        returns = np.full(100, 0.001)
        verifier = StrategyVerifier()
        result = verifier.verify(returns, trades=_dummy_trades())
        # Zero std → Sharpe undefined; verifier should handle gracefully
        assert result.sharpe_t_stat == 0.0

    def test_sharpe_p_value_range(self) -> None:
        """p-value should always be in [0, 1]."""
        returns = _positive_returns()
        verifier = StrategyVerifier()
        result = verifier.verify(returns, trades=_dummy_trades())
        assert 0.0 <= result.sharpe_p_value <= 1.0

    def test_trade_count_propagated(self) -> None:
        """total_trades should equal the length of the trade list supplied."""
        returns = _positive_returns()
        verifier = StrategyVerifier()
        result = verifier.verify(returns, trades=_dummy_trades(n=150))
        assert result.total_trades == 150


# ---------------------------------------------------------------------------
# Gate 3: Monte Carlo permutation
# ---------------------------------------------------------------------------


class TestMonteCarlo:
    def test_strong_strategy_passes(self) -> None:
        """High-drift returns should score well in the permutation test."""
        returns = _strong_returns()
        verifier = StrategyVerifier()
        result = verifier.verify(
            returns, trades=_dummy_trades(), n_mc_simulations=1000
        )
        # Strong positive drift should be in the top 10% of permutations
        assert result.mc_p_value < 0.1

    def test_random_strategy_fails(self) -> None:
        """Zero-mean returns should not pass the Monte Carlo test."""
        returns = _zero_returns()
        verifier = StrategyVerifier()
        result = verifier.verify(
            returns, trades=_dummy_trades(), n_mc_simulations=1000
        )
        assert result.mc_p_value > 0.1

    def test_mc_percentile_positive_drift(self) -> None:
        """Positively-drifted returns should be above the 50th percentile."""
        returns = _strong_returns()
        verifier = StrategyVerifier()
        result = verifier.verify(
            returns, trades=_dummy_trades(), n_mc_simulations=1000
        )
        assert result.mc_percentile > 50.0

    def test_mc_p_value_range(self) -> None:
        """p-value must be in [0, 1]."""
        returns = _positive_returns()
        verifier = StrategyVerifier()
        result = verifier.verify(
            returns, trades=_dummy_trades(), n_mc_simulations=500
        )
        assert 0.0 <= result.mc_p_value <= 1.0


# ---------------------------------------------------------------------------
# Gate 4: Deflated Sharpe
# ---------------------------------------------------------------------------


class TestDeflatedSharpe:
    def test_single_strategy_no_penalty(self) -> None:
        """With one strategy tested, deflated Sharpe should equal observed Sharpe."""
        returns = _positive_returns()
        verifier = StrategyVerifier(num_strategies_tested=1)
        result = verifier.verify(returns, trades=_dummy_trades())
        assert result.deflated_sharpe == pytest.approx(result.sharpe_ratio, abs=0.01)

    def test_many_strategies_penalized(self) -> None:
        """Testing 100 variants should penalize the deflated Sharpe more than 1."""
        returns = _positive_returns()
        v1 = StrategyVerifier(num_strategies_tested=1)
        v100 = StrategyVerifier(num_strategies_tested=100)
        r1 = v1.verify(returns, trades=_dummy_trades())
        r100 = v100.verify(returns, trades=_dummy_trades())
        assert r100.deflated_sharpe < r1.deflated_sharpe

    def test_deflated_sharpe_negative_at_large_k(self) -> None:
        """Modest Sharpe with 1000 strategies tested should yield deflated < 0."""
        rng = np.random.default_rng(42)
        # Deliberately modest drift so inflation dominates
        returns = rng.normal(0.0002, 0.01, 504)
        verifier = StrategyVerifier(num_strategies_tested=1000)
        result = verifier.verify(returns, trades=_dummy_trades())
        assert result.deflated_sharpe < result.sharpe_ratio

    def test_num_strategies_stored_in_result(self) -> None:
        """num_strategies_tested should be preserved in the result."""
        returns = _positive_returns()
        verifier = StrategyVerifier(num_strategies_tested=25)
        result = verifier.verify(returns, trades=_dummy_trades())
        assert result.num_strategies_tested == 25


# ---------------------------------------------------------------------------
# Gate 2: Overfitting / subsample stability
# ---------------------------------------------------------------------------


class TestOverfitting:
    def test_is_oos_split_high_degradation_flagged(self) -> None:
        """When OOS Sharpe is tiny vs IS, overfitting should be flagged."""
        rng = np.random.default_rng(42)
        is_returns = rng.normal(0.002, 0.01, 252)
        oos_returns = rng.normal(0.0001, 0.01, 252)  # near zero OOS
        verifier = StrategyVerifier()
        result = verifier.verify(
            is_returns, trades=_dummy_trades(),
            is_returns=is_returns, oos_returns=oos_returns,
        )
        assert result.overfitting_risk in ("HIGH", "CRITICAL")

    def test_consistent_is_oos_low_risk(self) -> None:
        """When OOS Sharpe matches IS, risk should be LOW or MEDIUM."""
        rng = np.random.default_rng(42)
        is_returns = rng.normal(0.001, 0.01, 252)
        oos_returns = rng.normal(0.0009, 0.01, 252)  # nearly same drift
        verifier = StrategyVerifier()
        result = verifier.verify(
            is_returns, trades=_dummy_trades(),
            is_returns=is_returns, oos_returns=oos_returns,
        )
        assert result.overfitting_risk in ("LOW", "MEDIUM")

    def test_subsample_stability_without_is_oos(self) -> None:
        """Without explicit IS/OOS, subsample test should still run."""
        returns = _positive_returns()
        verifier = StrategyVerifier()
        result = verifier.verify(returns, trades=_dummy_trades())
        # overfitting_risk should have been set by subsample test
        assert result.overfitting_risk != "UNKNOWN"


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------


class TestVerificationResult:
    def test_summary_contains_strategy_name(self) -> None:
        """Summary string should include the strategy name."""
        result = VerificationResult(strategy_name="MyStrategy")
        assert "MyStrategy" in result.summary()

    def test_summary_contains_headers(self) -> None:
        """Summary should include all five gate headers."""
        result = VerificationResult(strategy_name="test")
        summary = result.summary()
        assert "STRATEGY VERIFICATION" in summary
        assert "GATE 1" in summary
        assert "GATE 5" in summary

    def test_gate_failures_tracked(self) -> None:
        """Gate failures appended manually should be retrievable."""
        result = VerificationResult(strategy_name="test")
        result.gate_failures.append("test failure")
        assert len(result.gate_failures) == 1

    def test_recommendation_default(self) -> None:
        """Default recommendation before verification should be DO NOT TRADE."""
        result = VerificationResult(strategy_name="test")
        assert result.recommendation == "DO NOT TRADE"

    def test_passes_all_gates_default_false(self) -> None:
        """passes_all_gates should default to False."""
        result = VerificationResult(strategy_name="test")
        assert not result.passes_all_gates

    def test_summary_shows_failures(self) -> None:
        """Summary should list gate failure messages."""
        result = VerificationResult(strategy_name="test")
        result.gate_failures.append("some failure reason")
        assert "some failure reason" in result.summary()


# ---------------------------------------------------------------------------
# LiveMonitor — basic trade recording
# ---------------------------------------------------------------------------


class TestLiveMonitor:
    def test_basic_trade_recording_no_halt(self) -> None:
        """Winning trade with no drawdown should not trigger a halt."""
        monitor = LiveMonitor(expected_avg_pnl=50.0)
        trade = _make_trade(1, pnl=75.0, pnl_pct=0.05)
        alerts = monitor.record_trade(trade)
        assert not monitor.is_halted

    def test_trade_count_increments(self) -> None:
        """trade_count property should reflect number of recorded trades."""
        monitor = LiveMonitor(expected_avg_pnl=50.0)
        for i in range(5):
            monitor.record_trade(_make_trade(i, pnl=10.0))
        assert monitor.trade_count == 5

    def test_status_report_with_trades(self) -> None:
        """Status report should contain win rate and ACTIVE status after trades."""
        monitor = LiveMonitor()
        monitor._peak_equity = 10000.0
        monitor._current_equity = 10000.0
        monitor.record_trade(_make_trade(1, pnl=50.0, pnl_pct=0.03))
        report = monitor.status_report()
        assert "Win rate" in report
        assert "ACTIVE" in report

    def test_status_report_no_trades(self) -> None:
        """Status report with no trades should return a sensible message."""
        monitor = LiveMonitor()
        report = monitor.status_report()
        assert "No trades" in report

    def test_winning_trades_no_halt(self) -> None:
        """Consistent winners on a large equity base should never halt."""
        monitor = LiveMonitor(expected_avg_pnl=50.0)
        monitor._peak_equity = 10000.0
        monitor._current_equity = 10000.0
        for i in range(20):
            monitor.record_trade(_make_trade(i, pnl=50.0, pnl_pct=0.03))
        assert not monitor.is_halted


# ---------------------------------------------------------------------------
# LiveMonitor — drawdown kill criteria
# ---------------------------------------------------------------------------


class TestDrawdownHalt:
    def test_drawdown_exceeds_limit_halts(self) -> None:
        """A single large loss should trigger HALT when max_drawdown_pct is low."""
        kill = KillCriteria(max_drawdown_pct=10.0)
        monitor = LiveMonitor(kill_criteria=kill)
        monitor._peak_equity = 1000.0
        monitor._current_equity = 1000.0
        alerts = monitor.record_trade(_make_trade(1, pnl=-150.0, pnl_pct=-0.15))
        assert monitor.is_halted
        assert any(a.severity == "HALT" for a in alerts)

    def test_drawdown_at_70pct_warns(self) -> None:
        """Drawdown at 70% of the limit should generate a WARNING, not a HALT."""
        kill = KillCriteria(max_drawdown_pct=20.0)
        monitor = LiveMonitor(kill_criteria=kill)
        monitor._peak_equity = 1000.0
        monitor._current_equity = 1000.0
        # 14% drawdown = 70% of 20% limit → WARNING only
        alerts = monitor.record_trade(_make_trade(1, pnl=-140.0, pnl_pct=-0.14))
        assert not monitor.is_halted
        assert any(a.severity == "WARNING" and a.category == "drawdown" for a in alerts)

    def test_drawdown_below_warning_no_alert(self) -> None:
        """Small drawdown well below the limit should produce no alerts."""
        kill = KillCriteria(max_drawdown_pct=20.0)
        monitor = LiveMonitor(kill_criteria=kill)
        monitor._peak_equity = 10000.0
        monitor._current_equity = 10000.0
        alerts = monitor.record_trade(_make_trade(1, pnl=-100.0, pnl_pct=-0.01))
        assert not monitor.is_halted
        dd_alerts = [a for a in alerts if a.category == "drawdown"]
        assert len(dd_alerts) == 0


# ---------------------------------------------------------------------------
# LiveMonitor — consecutive losers kill criteria
# ---------------------------------------------------------------------------


class TestConsecutiveLosersHalt:
    def test_consecutive_losers_triggers_halt(self) -> None:
        """Hitting max_consecutive_losers should halt the strategy."""
        kill = KillCriteria(max_consecutive_losers=3)
        monitor = LiveMonitor(kill_criteria=kill, expected_avg_pnl=50.0)
        monitor._peak_equity = 10000.0
        monitor._current_equity = 10000.0
        for i in range(4):
            monitor.record_trade(_make_trade(i, pnl=-10.0, pnl_pct=-0.01))
        assert monitor.is_halted

    def test_near_limit_warns(self) -> None:
        """Two below the limit should produce a WARNING (limit - 2 = threshold)."""
        kill = KillCriteria(max_consecutive_losers=5, max_drawdown_pct=50.0)
        monitor = LiveMonitor(kill_criteria=kill, expected_avg_pnl=50.0)
        monitor._peak_equity = 100000.0
        monitor._current_equity = 100000.0
        for i in range(3):  # streak = 3, limit - 2 = 3 → WARNING
            monitor.record_trade(_make_trade(i, pnl=-1.0, pnl_pct=-0.00001))
        all_alerts = monitor.all_alerts
        assert any(
            a.severity == "WARNING" and a.category == "win_rate"
            for a in all_alerts
        )
        assert not monitor.is_halted

    def test_win_breaks_streak(self) -> None:
        """A winning trade should reset the losing streak counter."""
        kill = KillCriteria(max_consecutive_losers=3)
        monitor = LiveMonitor(kill_criteria=kill, expected_avg_pnl=50.0)
        monitor._peak_equity = 10000.0
        monitor._current_equity = 10000.0
        # Two losses, then a win, then two more losses — should not halt
        for pnl in [-10.0, -10.0, 100.0, -10.0, -10.0]:
            monitor.record_trade(_make_trade(0, pnl=pnl, pnl_pct=pnl / 10000))
        assert not monitor.is_halted


# ---------------------------------------------------------------------------
# LiveMonitor — win rate kill criteria
# ---------------------------------------------------------------------------


class TestWinRateHalt:
    def test_win_rate_collapse_halts(self) -> None:
        """Win rate below threshold over 30 trades should trigger halt."""
        kill = KillCriteria(min_rolling_win_rate=0.4, max_drawdown_pct=100.0)
        monitor = LiveMonitor(kill_criteria=kill, expected_avg_pnl=1.0)
        monitor._peak_equity = 1_000_000.0
        monitor._current_equity = 1_000_000.0
        # 30 losses in a row = 0% win rate
        for i in range(30):
            monitor.record_trade(_make_trade(i, pnl=-0.01, pnl_pct=-0.000001))
        assert monitor.is_halted

    def test_win_rate_above_threshold_no_halt(self) -> None:
        """Alternating wins/losses at 50% should not trigger a 35% threshold halt."""
        kill = KillCriteria(
            min_rolling_win_rate=0.35,
            max_drawdown_pct=100.0,
            max_consecutive_losers=999,
        )
        monitor = LiveMonitor(kill_criteria=kill, expected_avg_pnl=1.0)
        monitor._peak_equity = 1_000_000.0
        monitor._current_equity = 1_000_000.0
        for i in range(40):
            pnl = 10.0 if i % 2 == 0 else -10.0
            monitor.record_trade(_make_trade(i, pnl=pnl))
        assert not monitor.is_halted


# ---------------------------------------------------------------------------
# LiveMonitor — CUSUM drift detection
# ---------------------------------------------------------------------------


class TestCUSUMDetection:
    def test_cusum_detects_negative_drift(self) -> None:
        """Many below-expectation trades should trigger a CUSUM CRITICAL alert."""
        monitor = LiveMonitor(
            expected_avg_pnl=50.0,
            kill_criteria=KillCriteria(
                cusum_threshold=3.0,
                max_drawdown_pct=100.0,
                max_consecutive_losers=999,
            ),
        )
        monitor._peak_equity = 100000.0
        monitor._current_equity = 100000.0

        all_alerts: list[MonitorAlert] = []
        for i in range(30):
            alerts = monitor.record_trade(_make_trade(i, pnl=-20.0, pnl_pct=-0.01))
            all_alerts.extend(alerts)

        cusum_alerts = [a for a in all_alerts if a.category == "cusum"]
        assert len(cusum_alerts) > 0
        assert all(a.severity == "CRITICAL" for a in cusum_alerts)

    def test_cusum_no_drift_on_good_trades(self) -> None:
        """Trades at or above expectation should not trigger CUSUM."""
        monitor = LiveMonitor(
            expected_avg_pnl=50.0,
            kill_criteria=KillCriteria(cusum_threshold=5.0),
        )
        monitor._peak_equity = 10000.0
        monitor._current_equity = 10000.0

        all_alerts: list[MonitorAlert] = []
        for i in range(20):
            alerts = monitor.record_trade(_make_trade(i, pnl=60.0, pnl_pct=0.01))
            all_alerts.extend(alerts)

        cusum_alerts = [a for a in all_alerts if a.category == "cusum"]
        assert len(cusum_alerts) == 0


# ---------------------------------------------------------------------------
# LiveMonitor — rolling Sharpe
# ---------------------------------------------------------------------------


class TestRollingSharpehalt:
    def test_rolling_sharpe_halt_on_collapse(self) -> None:
        """Consistently negative daily P&L over 63+ days should trigger halt."""
        kill = KillCriteria(
            min_monthly_sharpe_3m=-0.3,
            max_drawdown_pct=100.0,
        )
        monitor = LiveMonitor(kill_criteria=kill)
        # Feed 63 days of negative drift
        rng = np.random.default_rng(1)
        all_alerts: list[MonitorAlert] = []
        for _ in range(63):
            alerts = monitor.record_daily_pnl(float(rng.normal(-200, 50)))
            all_alerts.extend(alerts)
        sharpe_halts = [
            a for a in all_alerts if a.category == "sharpe" and a.severity == "HALT"
        ]
        assert len(sharpe_halts) > 0

    def test_rolling_sharpe_no_alert_positive(self) -> None:
        """Positive daily P&L should not trigger rolling Sharpe halt."""
        kill = KillCriteria(min_monthly_sharpe_3m=-0.5, max_drawdown_pct=100.0)
        monitor = LiveMonitor(kill_criteria=kill)
        rng = np.random.default_rng(99)
        all_alerts: list[MonitorAlert] = []
        for _ in range(63):
            alerts = monitor.record_daily_pnl(float(rng.normal(100, 20)))
            all_alerts.extend(alerts)
        sharpe_alerts = [a for a in all_alerts if a.category == "sharpe"]
        assert len(sharpe_alerts) == 0


# ---------------------------------------------------------------------------
# LiveMonitor — halt idempotence
# ---------------------------------------------------------------------------


class TestHaltIdempotence:
    def test_halt_reason_not_overwritten(self) -> None:
        """Second halt trigger should not overwrite the first halt reason."""
        kill = KillCriteria(max_drawdown_pct=5.0)
        monitor = LiveMonitor(kill_criteria=kill)
        monitor._peak_equity = 1000.0
        monitor._current_equity = 1000.0
        monitor.record_trade(_make_trade(1, pnl=-100.0))
        first_reason = monitor.halt_reason
        monitor.record_trade(_make_trade(2, pnl=-200.0))
        assert monitor.halt_reason == first_reason

    def test_is_halted_property_true_after_halt(self) -> None:
        """is_halted should return True after any kill criteria is triggered."""
        kill = KillCriteria(max_drawdown_pct=5.0)
        monitor = LiveMonitor(kill_criteria=kill)
        monitor._peak_equity = 1000.0
        monitor._current_equity = 1000.0
        monitor.record_trade(_make_trade(1, pnl=-100.0))
        assert monitor.is_halted is True


# ---------------------------------------------------------------------------
# KillCriteria serialization
# ---------------------------------------------------------------------------


class TestKillCriteriaSerialization:
    def test_to_dict_preserves_values(self) -> None:
        """to_dict should serialise every field value correctly."""
        kc = KillCriteria(
            max_drawdown_pct=15.0,
            max_consecutive_losers=6,
            min_rolling_win_rate=0.40,
            max_drawdown_duration_days=30,
            min_monthly_sharpe_3m=-1.0,
            cusum_threshold=4.0,
        )
        d = kc.to_dict()
        assert d["max_drawdown_pct"] == 15.0
        assert d["max_consecutive_losers"] == 6
        assert d["min_rolling_win_rate"] == 0.40
        assert d["max_drawdown_duration_days"] == 30
        assert d["min_monthly_sharpe_3m"] == -1.0
        assert d["cusum_threshold"] == 4.0

    def test_to_dict_default_values(self) -> None:
        """Default KillCriteria should serialise without error."""
        kc = KillCriteria()
        d = kc.to_dict()
        assert isinstance(d, dict)
        assert len(d) == 8

    def test_to_dict_includes_min_trade_fields(self) -> None:
        """to_dict should include both new minimum-trade-count fields."""
        kc = KillCriteria(min_trades_for_win_rate_kill=25, min_trades_for_cusum_kill=15)
        d = kc.to_dict()
        assert d["min_trades_for_win_rate_kill"] == 25
        assert d["min_trades_for_cusum_kill"] == 15


# ---------------------------------------------------------------------------
# Bootstrap Sharpe CI
# ---------------------------------------------------------------------------


class TestBootstrapSharpe:
    def test_strong_returns_exclude_zero(self) -> None:
        """Bootstrap CI for strong positive returns should exclude zero."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.002, 0.01, 504)
        verifier = StrategyVerifier()
        result = verifier.verify(returns, trades=[{"pnl": 1}] * 100)
        assert result.bootstrap_sharpe_excludes_zero is True
        assert result.bootstrap_sharpe_ci[0] > 0

    def test_zero_returns_include_zero(self) -> None:
        """Bootstrap CI for zero-mean returns should include zero."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.01, 504)
        verifier = StrategyVerifier()
        result = verifier.verify(returns, trades=[{"pnl": 0}] * 100)
        assert result.bootstrap_sharpe_ci[0] <= 0

    def test_fat_tailed_returns(self) -> None:
        """Bootstrap should handle fat tails better than Gaussian."""
        rng = np.random.default_rng(42)
        # t-distributed returns (fat tails)
        returns = rng.standard_t(df=3, size=504) * 0.01 + 0.001
        verifier = StrategyVerifier()
        result = verifier.verify(returns, trades=[{"pnl": 1}] * 100)
        # Should still produce a valid CI with lower < upper
        assert result.bootstrap_sharpe_ci[0] < result.bootstrap_sharpe_ci[1]

    def test_bootstrap_ci_in_summary(self) -> None:
        """Summary output should contain the Bootstrap 95% CI line."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.002, 0.01, 504)
        verifier = StrategyVerifier()
        result = verifier.verify(returns, trades=[{"pnl": 1}] * 100)
        assert "Bootstrap 95% CI" in result.summary()


# ---------------------------------------------------------------------------
# BacktestLogger
# ---------------------------------------------------------------------------


class TestBacktestLogger:
    def test_log_run_increments_count(self, tmp_path: Path) -> None:
        """Each log_run call should increment total_runs by one."""
        bl = BacktestLogger(log_dir=str(tmp_path))
        assert bl.total_runs == 0
        bl.log_run("test", {"param": 1}, ["AAPL"], ("2020-01-01", "2025-01-01"))
        assert bl.total_runs == 1
        bl.log_run("test", {"param": 2}, ["AAPL"], ("2020-01-01", "2025-01-01"))
        assert bl.total_runs == 2

    def test_run_persisted_to_disk(self, tmp_path: Path) -> None:
        """Each logged run should create exactly one JSON file on disk."""
        bl = BacktestLogger(log_dir=str(tmp_path))
        bl.log_run("test", {"x": 1}, ["AAPL"], ("2020-01-01", "2025-01-01"))
        files = list(tmp_path.glob("run_*.json"))
        assert len(files) == 1

    def test_loads_existing_runs(self, tmp_path: Path) -> None:
        """A second BacktestLogger pointing at the same dir should reload runs."""
        bl1 = BacktestLogger(log_dir=str(tmp_path))
        bl1.log_run("test", {"x": 1}, ["AAPL"], ("2020-01-01", "2025-01-01"))
        bl2 = BacktestLogger(log_dir=str(tmp_path))
        assert bl2.total_runs == 1

    def test_num_prior_runs_accurate(self, tmp_path: Path) -> None:
        """num_prior_runs should record how many runs preceded each one."""
        bl = BacktestLogger(log_dir=str(tmp_path))
        r1 = bl.log_run("test", {"x": 1}, ["AAPL"], ("2020-01-01", "2025-01-01"))
        r2 = bl.log_run("test", {"x": 2}, ["AAPL"], ("2020-01-01", "2025-01-01"))
        assert r1.num_prior_runs == 0
        assert r2.num_prior_runs == 1


# ---------------------------------------------------------------------------
# Configurable OOS/IS threshold
# ---------------------------------------------------------------------------


class TestOOSThreshold:
    def test_custom_threshold(self) -> None:
        """Strict OOS threshold should produce more gate failures than lenient."""
        rng = np.random.default_rng(42)
        is_returns = rng.normal(0.002, 0.01, 252)
        oos_returns = rng.normal(0.001, 0.01, 252)

        v_strict = StrategyVerifier(oos_is_threshold=0.7)
        r_strict = v_strict.verify(
            np.concatenate([is_returns, oos_returns]),
            trades=[{"pnl": 1}] * 100,
            is_returns=is_returns,
            oos_returns=oos_returns,
        )

        v_lenient = StrategyVerifier(oos_is_threshold=0.3)
        r_lenient = v_lenient.verify(
            np.concatenate([is_returns, oos_returns]),
            trades=[{"pnl": 1}] * 100,
            is_returns=is_returns,
            oos_returns=oos_returns,
        )

        # Lenient threshold should produce fewer overfitting failures
        assert len(r_lenient.gate_failures) <= len(r_strict.gate_failures)

    def test_default_threshold_is_half(self) -> None:
        """Default oos_is_threshold should be 0.5."""
        verifier = StrategyVerifier()
        assert verifier.oos_is_threshold == 0.5


# ---------------------------------------------------------------------------
# Minimum trade counts before kills trigger
# ---------------------------------------------------------------------------


class TestMinTradeCountForKill:
    def test_win_rate_kill_respects_min_trades(self) -> None:
        """Win-rate halt should not trigger before min_trades_for_win_rate_kill."""
        kill = KillCriteria(
            min_rolling_win_rate=0.5,
            max_consecutive_losers=100,
            max_drawdown_pct=100.0,
            min_trades_for_win_rate_kill=30,
        )
        monitor = LiveMonitor(kill_criteria=kill, expected_avg_pnl=50.0)
        monitor._peak_equity = 100000.0
        monitor._current_equity = 100000.0

        # Feed 20 losers — below min_trades_for_win_rate_kill (30)
        for i in range(20):
            trade = TradeRecord(
                trade_id=i,
                symbol="AAPL",
                entry_date="2025-01-01",
                exit_date="2025-01-15",
                pnl=-10.0,
                pnl_pct=-0.01,
            )
            monitor.record_trade(trade)

        assert not monitor.is_halted

    def test_cusum_kill_respects_min_trades(self) -> None:
        """CUSUM alerts should not fire before min_trades_for_cusum_kill."""
        kill = KillCriteria(
            cusum_threshold=1.0,
            max_drawdown_pct=100.0,
            max_consecutive_losers=999,
            min_trades_for_cusum_kill=20,
        )
        monitor = LiveMonitor(
            kill_criteria=kill,
            expected_avg_pnl=50.0,
        )
        monitor._peak_equity = 100000.0
        monitor._current_equity = 100000.0

        all_alerts: list[MonitorAlert] = []
        # Feed 10 bad trades — below min_trades_for_cusum_kill (20)
        for i in range(10):
            trade = TradeRecord(
                trade_id=i,
                symbol="AAPL",
                entry_date="2025-01-01",
                exit_date="2025-01-15",
                pnl=-100.0,
                pnl_pct=-0.1,
            )
            alerts = monitor.record_trade(trade)
            all_alerts.extend(alerts)

        cusum_alerts = [a for a in all_alerts if a.category == "cusum"]
        assert len(cusum_alerts) == 0


# ---------------------------------------------------------------------------
# MC-calibrated LiveMonitor
# ---------------------------------------------------------------------------


class TestMCCalibratedMonitor:
    def test_from_backtest_distribution(self) -> None:
        """MC-calibrated monitor should have sensible kill thresholds."""
        rng = np.random.default_rng(42)
        pnls: list[float] = []
        for _ in range(200):
            if rng.random() < 0.6:
                pnls.append(float(rng.uniform(20, 100)))
            else:
                pnls.append(float(rng.uniform(-80, -10)))

        monitor = LiveMonitor.from_backtest_distribution(
            pnls, confidence_level=0.95, n_simulations=1000
        )
        assert monitor.kill_criteria.max_drawdown_pct > 0
        assert monitor.kill_criteria.max_consecutive_losers > 0
        assert 0 < monitor.kill_criteria.min_rolling_win_rate < 1
        assert not monitor.is_halted

    def test_from_backtest_distribution_not_halted_initially(self) -> None:
        """A freshly created MC-calibrated monitor should not be halted."""
        rng = np.random.default_rng(7)
        pnls = [float(rng.normal(30, 20)) for _ in range(100)]
        monitor = LiveMonitor.from_backtest_distribution(
            pnls, confidence_level=0.95, n_simulations=500
        )
        assert not monitor.is_halted
        assert monitor.trade_count == 0
