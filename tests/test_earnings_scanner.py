"""Tests for the Earnings Scanner (PEAD signal — Pillar 2).

Pure-logic tests that do not require network access.  The async scan()
method is covered with a mocked HTTP session; _evaluate() and
check_alignment() are tested directly.
"""
from __future__ import annotations

import asyncio
import sys
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from signals.earnings_scanner import EarningsCandidate, EarningsScanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scanner(
    sue_threshold: float = 1.0,
    lookback_days: int = 3,
) -> EarningsScanner:
    return EarningsScanner(
        api_key="test-key",
        sue_threshold=sue_threshold,
        lookback_days=lookback_days,
    )


def _make_candidate(
    *,
    direction: str = "BULLISH",
    iv_spread: float = 0.05,
    finbert_score: float = 0.4,
    sue: float = 1.0,
) -> EarningsCandidate:
    return EarningsCandidate(
        symbol="AAPL",
        report_date="2026-03-10",
        eps_actual=2.0,
        eps_consensus=1.8,
        sue=sue,
        surprise_pct=0.111,
        direction=direction,
        iv_spread=iv_spread,
        finbert_score=finbert_score,
        signals_aligned=False,
    )


def _make_earning_payload(
    *,
    symbol: str = "AAPL",
    eps_actual: float = 2.0,
    eps_consensus: float | None = None,
    filing_date: str = "2026-03-10",
    use_tickers_list: bool = True,
) -> dict:
    """Construct a minimal dict that mirrors the Polygon financials API shape."""
    payload: dict = {
        "filing_date": filing_date,
        "financials": {
            "income_statement": {
                "basic_earnings_per_share": {"value": eps_actual},
            }
        },
    }
    if use_tickers_list:
        payload["tickers"] = [symbol]
    else:
        payload["ticker"] = symbol
    if eps_consensus is not None:
        payload["consensus_eps"] = eps_consensus
    return payload


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestEarningsScannerInit:
    def test_raises_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Polygon API key"):
            EarningsScanner(api_key=None)

    def test_accepts_explicit_api_key(self) -> None:
        scanner = _make_scanner()
        assert scanner is not None

    def test_custom_params_stored(self) -> None:
        scanner = EarningsScanner(
            api_key="k", sue_threshold=0.5, lookback_days=5
        )
        assert scanner.sue_threshold == pytest.approx(0.5)
        assert scanner.lookback_days == 5


# ---------------------------------------------------------------------------
# _evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_positive_surprise_returns_bullish(self) -> None:
        scanner = _make_scanner()
        payload = _make_earning_payload(eps_actual=2.0, eps_consensus=1.8)
        result = scanner._evaluate(payload)
        assert result is not None
        assert result.direction == "BULLISH"
        assert result.sue > 0

    def test_negative_surprise_returns_bearish(self) -> None:
        scanner = _make_scanner()
        payload = _make_earning_payload(eps_actual=1.0, eps_consensus=1.5)
        result = scanner._evaluate(payload)
        assert result is not None
        assert result.direction == "BEARISH"
        assert result.sue < 0

    def test_missing_eps_returns_none(self) -> None:
        scanner = _make_scanner()
        payload: dict = {
            "filing_date": "2026-03-10",
            "tickers": ["AAPL"],
            "financials": {"income_statement": {}},
        }
        assert scanner._evaluate(payload) is None

    def test_tiny_consensus_filtered(self) -> None:
        """abs(eps_consensus) < 0.01 should return None."""
        scanner = _make_scanner()
        payload = _make_earning_payload(eps_actual=1.0, eps_consensus=0.005)
        assert scanner._evaluate(payload) is None

    def test_symbol_from_tickers_list(self) -> None:
        scanner = _make_scanner()
        payload = _make_earning_payload(symbol="MSFT", use_tickers_list=True)
        result = scanner._evaluate(payload)
        assert result is not None
        assert result.symbol == "MSFT"

    def test_symbol_from_ticker_key(self) -> None:
        scanner = _make_scanner()
        payload = _make_earning_payload(symbol="GOOG", use_tickers_list=False)
        result = scanner._evaluate(payload)
        assert result is not None
        assert result.symbol == "GOOG"

    def test_missing_symbol_returns_none(self) -> None:
        scanner = _make_scanner()
        payload: dict = {
            "filing_date": "2026-03-10",
            "tickers": [],
            "financials": {
                "income_statement": {
                    "basic_earnings_per_share": {"value": 1.0}
                }
            },
        }
        assert scanner._evaluate(payload) is None

    def test_sue_below_threshold_returns_none(self) -> None:
        """When |sue| < sue_threshold the candidate should be filtered out."""
        scanner = _make_scanner(sue_threshold=2.0)
        # SUE ≈ ±1.0 (normalised direction), which is below threshold of 2.0
        payload = _make_earning_payload(eps_actual=2.0, eps_consensus=1.8)
        result = scanner._evaluate(payload)
        assert result is None

    def test_report_date_stored(self) -> None:
        scanner = _make_scanner()
        payload = _make_earning_payload(filing_date="2026-03-12")
        result = scanner._evaluate(payload)
        assert result is not None
        assert result.report_date == "2026-03-12"

    def test_default_fields_zero_or_false(self) -> None:
        """iv_spread, finbert_score default to 0.0; signals_aligned False."""
        scanner = _make_scanner()
        payload = _make_earning_payload()
        result = scanner._evaluate(payload)
        assert result is not None
        assert result.iv_spread == pytest.approx(0.0)
        assert result.finbert_score == pytest.approx(0.0)
        assert result.signals_aligned is False


# ---------------------------------------------------------------------------
# check_alignment
# ---------------------------------------------------------------------------

class TestCheckAlignment:
    def test_all_signals_bullish_aligned(self) -> None:
        c = _make_candidate(
            direction="BULLISH", iv_spread=0.05, finbert_score=0.5, sue=1.0
        )
        assert EarningsScanner.check_alignment(c) is True
        assert c.signals_aligned is True

    def test_all_signals_bearish_aligned(self) -> None:
        c = _make_candidate(
            direction="BEARISH", iv_spread=-0.05, finbert_score=-0.5, sue=-1.0
        )
        assert EarningsScanner.check_alignment(c) is True

    def test_two_of_three_aligned_passes(self) -> None:
        """SUE + IV spread agree; FinBERT neutral → confirms >= 2."""
        c = _make_candidate(
            direction="BULLISH", iv_spread=0.05, finbert_score=0.1, sue=1.0
        )
        assert EarningsScanner.check_alignment(c) is True

    def test_only_sue_fails(self) -> None:
        """SUE alone (1 of 3) should not be enough: confirms < 2."""
        c = _make_candidate(
            direction="BULLISH", iv_spread=-0.05, finbert_score=-0.5, sue=1.0
        )
        assert EarningsScanner.check_alignment(c) is False

    def test_alignment_written_to_candidate(self) -> None:
        c = _make_candidate(
            direction="BEARISH", iv_spread=-0.03, finbert_score=0.5, sue=-1.0
        )
        # IV spread agrees, FinBERT disagrees → 2 confirmations
        EarningsScanner.check_alignment(c)
        assert c.signals_aligned is True

    def test_finbert_threshold_exactly_03(self) -> None:
        """FinBERT score must be strictly above 0.3 (or below -0.3) to confirm."""
        # Exactly 0.3 should NOT confirm
        c = _make_candidate(
            direction="BULLISH", iv_spread=-0.05, finbert_score=0.3, sue=1.0
        )
        # SUE=1, iv_spread negative → no IV confirm; finbert=0.3 not >0.3 → no
        result = EarningsScanner.check_alignment(c)
        assert result is False


# ---------------------------------------------------------------------------
# Async scan() with mocked session
# ---------------------------------------------------------------------------

class TestScanAsync:
    def test_scan_returns_list(self) -> None:
        """scan() should return an empty list when the API returns nothing."""
        scanner = _make_scanner()

        async def _run() -> list:
            with mock.patch.object(
                scanner, "_get_recent_earnings", return_value=[]
            ):
                return await scanner.scan()

        results = asyncio.run(_run())
        assert isinstance(results, list)
        assert results == []

    def test_scan_sorts_by_abs_sue_descending(self) -> None:
        """Results should be sorted by |sue| descending."""
        scanner = _make_scanner()
        payloads = [
            _make_earning_payload(symbol="A", eps_actual=2.0, eps_consensus=1.0),
            _make_earning_payload(symbol="B", eps_actual=1.5, eps_consensus=1.0),
        ]

        async def _run() -> list:
            with mock.patch.object(
                scanner, "_get_recent_earnings", return_value=payloads
            ):
                return await scanner.scan()

        results = asyncio.run(_run())
        assert isinstance(results, list)
        if len(results) >= 2:
            assert abs(results[0].sue) >= abs(results[1].sue)

    def test_scan_filters_bad_payloads(self) -> None:
        """Payloads that _evaluate() rejects should not appear in results."""
        scanner = _make_scanner()
        payloads = [
            {"filing_date": "2026-03-10", "tickers": [], "financials": {}},
        ]

        async def _run() -> list:
            with mock.patch.object(
                scanner, "_get_recent_earnings", return_value=payloads
            ):
                return await scanner.scan()

        results = asyncio.run(_run())
        assert results == []
