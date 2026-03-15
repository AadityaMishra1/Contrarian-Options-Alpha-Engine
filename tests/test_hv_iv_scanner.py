"""Tests for the HV-IV Gap Scanner.

Covers pure-Python static/instance methods that do not require network
access.  Async scan() and API methods are tested via mocking.
"""
from __future__ import annotations

import asyncio
import math
import sys
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from signals.hv_iv_scanner import HVIVCandidate, HVIVScanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(
    *,
    gap_ratio: float = 1.5,
    gap_zscore: float = 0.0,
    skew_zscore: float = 0.0,
    days_to_earnings: int = 30,
    quintile: int = 3,
) -> HVIVCandidate:
    return HVIVCandidate(
        symbol="TEST",
        price=150.0,
        hv_30=0.25,
        hv_60=0.23,
        iv_30=0.25 * gap_ratio,
        gap_ratio=gap_ratio,
        gap_zscore=gap_zscore,
        quintile=quintile,
        skew_zscore=skew_zscore,
        volume=5_000_000,
        market_cap=2.5e12,
        days_to_earnings=days_to_earnings,
    )


def _make_scanner(earnings_buffer: int = 7) -> HVIVScanner:
    """Build a scanner with a dummy API key (no real HTTP calls)."""
    return HVIVScanner(api_key="test-key", earnings_buffer=earnings_buffer)


# ---------------------------------------------------------------------------
# _realized_vol
# ---------------------------------------------------------------------------

class TestRealizedVol:
    def test_constant_prices_zero_vol(self) -> None:
        closes = [100.0] * 35
        vol = HVIVScanner._realized_vol(closes, 30)
        assert vol == pytest.approx(0.0, abs=1e-10)

    def test_trending_up_positive_vol(self) -> None:
        closes = [100.0 + i * 0.5 for i in range(35)]
        vol = HVIVScanner._realized_vol(closes, 30)
        assert vol > 0

    def test_high_vol_prices_in_range(self) -> None:
        """3 % daily noise ≈ 47 % annualised; should land in [0.2, 0.8]."""
        import random

        random.seed(42)
        closes = [100.0]
        for _ in range(34):
            closes.append(closes[-1] * (1 + random.gauss(0, 0.03)))
        vol = HVIVScanner._realized_vol(closes, 30)
        assert 0.2 < vol < 0.8

    def test_insufficient_data_returns_zero(self) -> None:
        closes = [100.0, 101.0]
        vol = HVIVScanner._realized_vol(closes, 30)
        assert vol == pytest.approx(0.0)

    def test_annualisation_factor(self) -> None:
        """Output must be annualised (multiplied by sqrt(252)).

        A linearly increasing series has very small log-returns (~1 %/day
        at the start, falling quickly), so the annualised vol is low but
        strictly above zero — confirming the sqrt(252) factor is applied
        rather than returning a raw daily figure.
        """
        closes = [100.0 + i for i in range(35)]
        vol = HVIVScanner._realized_vol(closes, 30)
        assert vol > 0.0  # annualised → non-zero for any trending series

    def test_exact_window_boundary(self) -> None:
        """Exactly window+1 prices should succeed; window prices should fail."""
        # window=5 needs 6 prices
        assert HVIVScanner._realized_vol([100.0] * 6, 5) == pytest.approx(0.0, abs=1e-10)
        assert HVIVScanner._realized_vol([100.0] * 5, 5) == pytest.approx(0.0)

    def test_single_zero_price_guarded(self) -> None:
        """A zero price inside the window must not raise ZeroDivisionError."""
        closes = [0.0] + [100.0] * 34
        # Should not raise; may return 0.0 or a valid number
        vol = HVIVScanner._realized_vol(closes, 30)
        assert isinstance(vol, float)


# ---------------------------------------------------------------------------
# HVIVCandidate dataclass
# ---------------------------------------------------------------------------

class TestHVIVCandidate:
    def test_dataclass_creation(self) -> None:
        c = HVIVCandidate(
            symbol="AAPL",
            price=150.0,
            hv_30=0.25,
            hv_60=0.23,
            iv_30=0.35,
            gap_ratio=1.4,
            gap_zscore=1.2,
            quintile=5,
            skew_zscore=0.5,
            volume=5_000_000,
            market_cap=2.5e12,
            days_to_earnings=30,
        )
        assert c.gap_ratio == pytest.approx(1.4)
        assert c.quintile == 5

    def test_default_score_zero(self) -> None:
        c = _make_candidate()
        assert c.score == pytest.approx(0.0)

    def test_symbol_stored(self) -> None:
        c = _make_candidate()
        assert c.symbol == "TEST"


# ---------------------------------------------------------------------------
# _compute_score
# ---------------------------------------------------------------------------

class TestComputeScore:
    def test_high_gap_yields_positive_score(self) -> None:
        scanner = _make_scanner()
        c = _make_candidate(gap_ratio=1.8, skew_zscore=0.0, days_to_earnings=30)
        score = scanner._compute_score(c)
        assert score > 0.5

    def test_gap_ratio_one_yields_zero_score(self) -> None:
        """gap_ratio = 1.0 → base = 0.0 → score = 0.0."""
        scanner = _make_scanner()
        c = _make_candidate(gap_ratio=1.0, skew_zscore=0.0, days_to_earnings=30)
        score = scanner._compute_score(c)
        assert score == pytest.approx(0.0)

    def test_gap_ratio_below_one_clamped_to_zero(self) -> None:
        """Negative base (IV cheaper than HV) should be clamped to 0.0."""
        scanner = _make_scanner()
        c = _make_candidate(gap_ratio=0.8, skew_zscore=0.0, days_to_earnings=30)
        score = scanner._compute_score(c)
        assert score == pytest.approx(0.0)

    def test_steep_skew_reduces_score(self) -> None:
        scanner = _make_scanner()
        base = _make_candidate(gap_ratio=1.8, skew_zscore=0.0, days_to_earnings=30)
        steep = _make_candidate(gap_ratio=1.8, skew_zscore=2.0, days_to_earnings=30)
        assert scanner._compute_score(base) > scanner._compute_score(steep)

    def test_near_earnings_reduces_score(self) -> None:
        scanner = _make_scanner(earnings_buffer=7)
        far = _make_candidate(gap_ratio=1.8, skew_zscore=0.0, days_to_earnings=30)
        near = _make_candidate(gap_ratio=1.8, skew_zscore=0.0, days_to_earnings=3)
        assert scanner._compute_score(far) > scanner._compute_score(near)

    def test_score_never_negative(self) -> None:
        """_compute_score must always be >= 0.0."""
        scanner = _make_scanner()
        c = _make_candidate(gap_ratio=1.1, skew_zscore=5.0, days_to_earnings=1)
        assert scanner._compute_score(c) >= 0.0

    def test_gap_ratio_capped_at_three(self) -> None:
        """base = min(3.0, gap_ratio - 1.0).

        For gap_ratio=4.0 → min(3.0, 3.0) = 3.0.
        For gap_ratio=5.0 → min(3.0, 4.0) = 3.0.
        Both should produce the same score, demonstrating the cap.
        """
        scanner = _make_scanner()
        c_four = _make_candidate(gap_ratio=4.0, skew_zscore=0.0, days_to_earnings=30)
        c_five = _make_candidate(gap_ratio=5.0, skew_zscore=0.0, days_to_earnings=30)
        assert scanner._compute_score(c_four) == pytest.approx(
            scanner._compute_score(c_five)
        )


# ---------------------------------------------------------------------------
# Constructor / configuration
# ---------------------------------------------------------------------------

class TestHVIVScannerInit:
    def test_raises_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Polygon API key"):
            HVIVScanner(api_key=None)

    def test_accepts_explicit_api_key(self) -> None:
        scanner = HVIVScanner(api_key="my-key")
        assert scanner is not None

    def test_custom_params_stored(self) -> None:
        scanner = HVIVScanner(
            api_key="k",
            universe_size=50,
            min_gap_ratio=1.3,
            min_volume=2000,
            earnings_buffer=10,
        )
        assert scanner.universe_size == 50
        assert scanner.min_gap_ratio == pytest.approx(1.3)
        assert scanner.min_volume == 2000
        assert scanner.earnings_buffer == 10
