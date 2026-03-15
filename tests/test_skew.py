"""Tests for the IV Skew Tracker."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from signals.skew import SkewReading, SkewTracker


class TestSkewTracker:
    def test_first_reading_zscore_zero(self) -> None:
        """With fewer than 10 observations the z-score is always 0.0."""
        st = SkewTracker()
        reading = st.update("AAPL", iv_otm_put=0.35, iv_atm=0.30)
        assert reading.smirk == pytest.approx(0.05)
        assert reading.zscore == pytest.approx(0.0)

    def test_smirk_computation(self) -> None:
        st = SkewTracker()
        reading = st.update("AAPL", iv_otm_put=0.40, iv_atm=0.25)
        assert reading.smirk == pytest.approx(0.15)

    def test_steep_detection_after_history(self) -> None:
        st = SkewTracker(steep_threshold=1.5)
        # Build stable history (20 normal observations)
        for _ in range(20):
            st.update("AAPL", iv_otm_put=0.35, iv_atm=0.30)
        # Sudden steep skew significantly above baseline
        reading = st.update("AAPL", iv_otm_put=0.50, iv_atm=0.30)
        assert reading.is_steep is True

    def test_flat_detection(self) -> None:
        st = SkewTracker(flat_threshold=-1.0)
        # Build history at a normal elevated smirk
        for _ in range(20):
            st.update("AAPL", iv_otm_put=0.35, iv_atm=0.30)
        # Flat / inverted skew — OTM puts priced same as ATM
        reading = st.update("AAPL", iv_otm_put=0.30, iv_atm=0.30)
        assert reading.is_flat is True

    def test_position_scalar_flat_skew(self) -> None:
        """Flat skew (z < flat_threshold) should yield scalar == 1.0."""
        st = SkewTracker()
        for _ in range(20):
            st.update("AAPL", iv_otm_put=0.35, iv_atm=0.30)
        st.update("AAPL", iv_otm_put=0.30, iv_atm=0.30)
        scalar = st.get_position_scalar("AAPL")
        assert scalar >= 0.8

    def test_position_scalar_steep_skew(self) -> None:
        """Very steep skew should heavily discount position size."""
        st = SkewTracker()
        for _ in range(20):
            st.update("AAPL", iv_otm_put=0.35, iv_atm=0.30)
        # Push skew very high (z > 2.5) to trigger 0.0 / 0.25 range
        st.update("AAPL", iv_otm_put=0.55, iv_atm=0.30)
        scalar = st.get_position_scalar("AAPL")
        assert scalar <= 0.5

    def test_position_scalar_unknown_symbol(self) -> None:
        """Default scalar for symbols without history is 0.8."""
        st = SkewTracker()
        scalar = st.get_position_scalar("UNKNOWN")
        assert scalar == pytest.approx(0.8)

    def test_reset_single_symbol(self) -> None:
        st = SkewTracker()
        for _ in range(15):
            st.update("AAPL", 0.35, 0.30)
        for _ in range(15):
            st.update("MSFT", 0.32, 0.28)
        st.reset("AAPL")
        assert st.get_position_scalar("AAPL") == pytest.approx(0.8)
        # MSFT history still intact — scalar differs from default 0.8
        # (it has 15 observations of a constant smirk → z=0 → scalar=0.9)
        assert st.get_position_scalar("MSFT") != pytest.approx(0.8)

    def test_reset_all(self) -> None:
        st = SkewTracker()
        for _ in range(15):
            st.update("AAPL", 0.35, 0.30)
        for _ in range(15):
            st.update("MSFT", 0.32, 0.28)
        st.reset()
        assert st.get_position_scalar("AAPL") == pytest.approx(0.8)
        assert st.get_position_scalar("MSFT") == pytest.approx(0.8)

    def test_multiple_symbols_independent(self) -> None:
        st = SkewTracker()
        st.update("AAPL", 0.35, 0.30)
        st.update("MSFT", 0.40, 0.28)
        r1 = st.update("AAPL", 0.36, 0.30)
        r2 = st.update("MSFT", 0.41, 0.28)
        assert r1.symbol == "AAPL"
        assert r2.symbol == "MSFT"
        assert r1.smirk != r2.smirk

    def test_skew_reading_fields(self) -> None:
        st = SkewTracker()
        reading = st.update("SPY", iv_otm_put=0.22, iv_atm=0.18)
        assert isinstance(reading, SkewReading)
        assert reading.symbol == "SPY"
        assert reading.iv_otm_put == pytest.approx(0.22)
        assert reading.iv_atm == pytest.approx(0.18)
        assert reading.smirk == pytest.approx(0.04)
        assert isinstance(reading.is_steep, bool)
        assert isinstance(reading.is_flat, bool)

    def test_zscore_computed_after_ten_observations(self) -> None:
        """Z-score should remain 0.0 below 10 obs and vary above 10."""
        st = SkewTracker()
        for i in range(9):
            r = st.update("X", iv_otm_put=0.30, iv_atm=0.25)
            assert r.zscore == pytest.approx(0.0), f"Expected 0.0 at obs {i+1}"
        # 10th observation — still exactly the mean, so z should be 0.0
        r10 = st.update("X", iv_otm_put=0.30, iv_atm=0.25)
        assert r10.zscore == pytest.approx(0.0)

    def test_lookback_window_respected(self) -> None:
        """History deque should not exceed the configured lookback size."""
        st = SkewTracker(lookback=10)
        for i in range(25):
            st.update("Z", iv_otm_put=0.30 + i * 0.001, iv_atm=0.25)
        # pylint: disable=protected-access
        assert len(st._history["Z"]) <= 10
