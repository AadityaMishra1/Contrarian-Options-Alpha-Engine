"""Tests for the VIX/VVIX Regime Classifier."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from signals.regime import Regime, RegimeClassifier, RegimeState


class TestRegimeClassifier:
    def test_crisis_regime(self) -> None:
        rc = RegimeClassifier()
        state = rc.classify(vix=40.0, vvix=130.0, vvix_5d_ago=120.0)
        assert state.regime == Regime.CRISIS
        assert state.position_scalar == 0.0

    def test_elevated_regime_vvix_rising(self) -> None:
        rc = RegimeClassifier()
        state = rc.classify(vix=28.0, vvix=110.0, vvix_5d_ago=100.0)
        assert state.regime == Regime.ELEVATED
        assert state.position_scalar == 0.5

    def test_recovery_regime_vvix_falling(self) -> None:
        rc = RegimeClassifier()
        state = rc.classify(vix=28.0, vvix=100.0, vvix_5d_ago=110.0)
        assert state.regime == Regime.RECOVERY
        assert state.position_scalar == 1.0

    def test_normal_regime(self) -> None:
        rc = RegimeClassifier()
        state = rc.classify(vix=20.0, vvix=90.0, vvix_5d_ago=88.0)
        assert state.regime == Regime.NORMAL
        assert state.position_scalar == 1.0

    def test_quiet_regime_low_vvix(self) -> None:
        """VIX below normal_vix with VVIX below quiet_vvix threshold (85)."""
        rc = RegimeClassifier()
        state = rc.classify(vix=14.0, vvix=75.0, vvix_5d_ago=74.0)
        assert state.regime == Regime.QUIET
        assert state.position_scalar == 0.6

    def test_quiet_regime_high_vvix(self) -> None:
        """VIX below normal_vix with VVIX above quiet_vvix threshold (85)."""
        rc = RegimeClassifier()
        state = rc.classify(vix=14.0, vvix=90.0, vvix_5d_ago=88.0)
        assert state.regime == Regime.QUIET
        assert state.position_scalar == 0.75

    def test_is_safe_to_sell_normal(self) -> None:
        rc = RegimeClassifier()
        rc.classify(vix=20.0, vvix=90.0, vvix_5d_ago=88.0)
        assert rc.is_safe_to_sell is True

    def test_is_safe_to_sell_crisis(self) -> None:
        rc = RegimeClassifier()
        rc.classify(vix=45.0, vvix=150.0, vvix_5d_ago=140.0)
        assert rc.is_safe_to_sell is False

    def test_boundary_vix_25(self) -> None:
        """VIX exactly at 25 should be NORMAL, not ELEVATED.

        The condition for ELEVATED/RECOVERY is vix > elevated_vix (strictly),
        so vix=25 with default elevated_vix=25 falls through to the NORMAL
        branch (vix >= normal_vix).
        """
        rc = RegimeClassifier()
        state = rc.classify(vix=25.0, vvix=90.0, vvix_5d_ago=88.0)
        assert state.regime == Regime.NORMAL

    def test_boundary_vix_35(self) -> None:
        """VIX exactly at 35 should be ELEVATED/RECOVERY, not CRISIS.

        CRISIS requires vix > crisis_vix (strictly greater), so vix=35 with
        default crisis_vix=35 lands in the elevated/recovery branch.
        """
        rc = RegimeClassifier()
        state = rc.classify(vix=35.0, vvix=100.0, vvix_5d_ago=105.0)
        assert state.regime == Regime.RECOVERY

    def test_custom_crisis_threshold(self) -> None:
        """Custom crisis_vix should override the default 35.0."""
        rc = RegimeClassifier(crisis_vix=30.0, elevated_vix=20.0)
        state = rc.classify(vix=32.0, vvix=100.0, vvix_5d_ago=90.0)
        assert state.regime == Regime.CRISIS

    def test_history_maintained(self) -> None:
        rc = RegimeClassifier()
        rc.classify(vix=20.0, vvix=90.0, vvix_5d_ago=88.0)
        rc.classify(vix=30.0, vvix=110.0, vvix_5d_ago=100.0)
        assert rc.current is not None
        assert rc.current.regime == Regime.ELEVATED

    def test_current_none_before_classify(self) -> None:
        rc = RegimeClassifier()
        assert rc.current is None
        assert rc.is_safe_to_sell is False

    def test_regime_state_fields(self) -> None:
        """RegimeState carries the raw vix/vvix values and a description."""
        rc = RegimeClassifier()
        state = rc.classify(vix=20.0, vvix=92.0, vvix_5d_ago=89.0)
        assert state.vix == pytest.approx(20.0)
        assert state.vvix == pytest.approx(92.0)
        assert state.vvix_5d_change == pytest.approx(3.0)
        assert isinstance(state.description, str)
        assert len(state.description) > 0

    def test_vvix_5d_change_sign(self) -> None:
        """vvix_5d_change sign determines ELEVATED vs RECOVERY direction."""
        rc = RegimeClassifier()
        # Rising VVIX
        state_up = rc.classify(vix=30.0, vvix=105.0, vvix_5d_ago=100.0)
        assert state_up.vvix_5d_change > 0
        assert state_up.regime == Regime.ELEVATED
        # Falling VVIX
        state_down = rc.classify(vix=30.0, vvix=95.0, vvix_5d_ago=100.0)
        assert state_down.vvix_5d_change < 0
        assert state_down.regime == Regime.RECOVERY

    def test_history_capped_at_252(self) -> None:
        """Internal history should not grow beyond 252 entries."""
        rc = RegimeClassifier()
        for _ in range(300):
            rc.classify(vix=20.0, vvix=90.0, vvix_5d_ago=88.0)
        # pylint: disable=protected-access
        assert len(rc._history) <= 252
