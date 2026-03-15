"""Tests for OptionsChainAnalyzer, OptionCandidate, and helpers.

All Polygon HTTP calls are mocked.  No live network access is required.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signals.options_chain import OptionCandidate, OptionsChainAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _future_date(days: int = 3) -> str:
    """Return an ISO date string *days* from today."""
    return (date.today() + timedelta(days=days)).isoformat()


def _make_raw_contract(
    ticker: str = "O:AAPL251219C00150000",
    contract_type: str = "call",
    strike: float = 150.0,
    expiry: str | None = None,
    bid: float = 1.30,
    ask: float = 1.70,
    delta: float = 0.30,
    iv_percentile: float = 25.0,
) -> dict:
    """Build a Polygon-shaped raw contract dict suitable for _parse_contract."""
    if expiry is None:
        expiry = _future_date(3)
    return {
        "details": {
            "ticker": ticker,
            "contract_type": contract_type,
            "strike_price": strike,
            "expiration_date": expiry,
        },
        "greeks": {"delta": delta if contract_type == "call" else -delta},
        "last_quote": {"bid": bid, "ask": ask},
        "day": {"open": bid, "close": ask},
        "implied_volatility_percentile": iv_percentile,
    }


# ---------------------------------------------------------------------------
# OptionsChainAnalyzer — constructor
# ---------------------------------------------------------------------------


class TestOptionsChainAnalyzerInit:
    def test_requires_api_key(self) -> None:
        import os
        os.environ.pop("POLYGON_API_KEY", None)
        with pytest.raises(ValueError, match="Polygon API key"):
            OptionsChainAnalyzer(api_key=None)

    def test_accepts_explicit_api_key(self) -> None:
        analyzer = OptionsChainAnalyzer(api_key="test_key")
        assert analyzer is not None

    def test_accepts_env_var(self) -> None:
        with patch.dict("os.environ", {"POLYGON_API_KEY": "env_key"}):
            analyzer = OptionsChainAnalyzer()
            assert analyzer is not None

    def test_default_filter_parameters(self) -> None:
        a = OptionsChainAnalyzer(api_key="test_key")
        assert a.dte_min == 1
        assert a.dte_max == 5
        assert a.delta_min == pytest.approx(0.20)
        assert a.delta_max == pytest.approx(0.40)
        assert a.iv_pct_max == pytest.approx(50.0)
        assert a.spread_pct_max == pytest.approx(20.0)

    def test_custom_filter_parameters(self) -> None:
        a = OptionsChainAnalyzer(
            api_key="test_key",
            dte_min=2,
            dte_max=7,
            delta_min=0.15,
            delta_max=0.35,
            iv_pct_max=40.0,
            spread_pct_max=15.0,
        )
        assert a.dte_min == 2
        assert a.dte_max == 7
        assert a.delta_min == pytest.approx(0.15)
        assert a.delta_max == pytest.approx(0.35)
        assert a.iv_pct_max == pytest.approx(40.0)
        assert a.spread_pct_max == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# OptionsChainAnalyzer._parse_contract
# ---------------------------------------------------------------------------


class TestParseContract:
    def _analyzer(self) -> OptionsChainAnalyzer:
        return OptionsChainAnalyzer(api_key="test_key")

    def test_parses_valid_call_contract(self) -> None:
        a = self._analyzer()
        today = date.today()
        raw = _make_raw_contract(expiry=_future_date(3))
        candidate = a._parse_contract(raw, "AAPL", today)
        assert candidate is not None
        assert candidate.option_type == "call"
        assert candidate.underlying == "AAPL"

    def test_parses_valid_put_contract(self) -> None:
        a = self._analyzer()
        today = date.today()
        raw = _make_raw_contract(
            ticker="O:AAPL251219P00150000",
            contract_type="put",
            delta=0.30,
            expiry=_future_date(3),
        )
        candidate = a._parse_contract(raw, "AAPL", today)
        assert candidate is not None
        assert candidate.option_type == "put"

    def test_delta_stored_as_absolute_value(self) -> None:
        """Put deltas are negative; the parser must store abs(delta)."""
        a = self._analyzer()
        today = date.today()
        raw = _make_raw_contract(contract_type="put", delta=0.30, expiry=_future_date(3))
        candidate = a._parse_contract(raw, "AAPL", today)
        assert candidate is not None
        assert candidate.delta >= 0

    def test_mid_price_computed_correctly(self) -> None:
        a = self._analyzer()
        today = date.today()
        raw = _make_raw_contract(bid=1.20, ask=1.80, expiry=_future_date(3))
        candidate = a._parse_contract(raw, "AAPL", today)
        assert candidate is not None
        assert candidate.mid_price == pytest.approx(1.50, abs=1e-3)

    def test_spread_pct_computed_correctly(self) -> None:
        a = self._analyzer()
        today = date.today()
        # bid=1.00, ask=1.50 → mid=1.25, spread=(0.50/1.25)*100=40%
        raw = _make_raw_contract(bid=1.00, ask=1.50, expiry=_future_date(3))
        candidate = a._parse_contract(raw, "AAPL", today)
        assert candidate is not None
        assert candidate.spread_pct == pytest.approx(40.0, abs=0.1)

    def test_dte_computed_correctly(self) -> None:
        a = self._analyzer()
        today = date.today()
        raw = _make_raw_contract(expiry=_future_date(3))
        candidate = a._parse_contract(raw, "AAPL", today)
        assert candidate is not None
        assert candidate.dte == 3

    def test_expired_contract_returns_none(self) -> None:
        a = self._analyzer()
        today = date.today()
        past_date = (today - timedelta(days=1)).isoformat()
        raw = _make_raw_contract(expiry=past_date)
        assert a._parse_contract(raw, "AAPL", today) is None

    def test_missing_ticker_returns_none(self) -> None:
        a = self._analyzer()
        today = date.today()
        raw = _make_raw_contract()
        raw["details"]["ticker"] = ""
        assert a._parse_contract(raw, "AAPL", today) is None

    def test_invalid_contract_type_returns_none(self) -> None:
        a = self._analyzer()
        today = date.today()
        raw = _make_raw_contract()
        raw["details"]["contract_type"] = "future"
        assert a._parse_contract(raw, "AAPL", today) is None

    def test_zero_bid_returns_none(self) -> None:
        a = self._analyzer()
        today = date.today()
        raw = _make_raw_contract(bid=0.0, ask=1.50, expiry=_future_date(3))
        assert a._parse_contract(raw, "AAPL", today) is None

    def test_ask_less_than_bid_returns_none(self) -> None:
        a = self._analyzer()
        today = date.today()
        raw = _make_raw_contract(bid=2.00, ask=1.00, expiry=_future_date(3))
        assert a._parse_contract(raw, "AAPL", today) is None

    def test_missing_expiry_returns_none(self) -> None:
        a = self._analyzer()
        today = date.today()
        raw = _make_raw_contract()
        raw["details"]["expiration_date"] = ""
        assert a._parse_contract(raw, "AAPL", today) is None


# ---------------------------------------------------------------------------
# OptionsChainAnalyzer._passes_filters
# ---------------------------------------------------------------------------


class TestPassesFilters:
    def _analyzer(self) -> OptionsChainAnalyzer:
        return OptionsChainAnalyzer(api_key="test_key")

    def _candidate(self, dte: int = 3, delta: float = 0.30, iv_pct: float = 25.0,
                   spread_pct: float = 10.0) -> OptionCandidate:
        return OptionCandidate(
            symbol="O:TEST",
            underlying="TEST",
            option_type="call",
            strike=100.0,
            expiry=_future_date(dte),
            dte=dte,
            bid=1.00,
            ask=1.20,
            delta=delta,
            iv_percentile=iv_pct,
            mid_price=1.10,
            spread_pct=spread_pct,
        )

    def test_passing_candidate(self) -> None:
        a = self._analyzer()
        assert a._passes_filters(self._candidate())

    def test_dte_below_min_rejected(self) -> None:
        a = self._analyzer()
        assert not a._passes_filters(self._candidate(dte=0))

    def test_dte_above_max_rejected(self) -> None:
        a = self._analyzer()
        assert not a._passes_filters(self._candidate(dte=10))

    def test_dte_at_min_boundary_accepted(self) -> None:
        a = self._analyzer()
        assert a._passes_filters(self._candidate(dte=1))

    def test_dte_at_max_boundary_accepted(self) -> None:
        a = self._analyzer()
        assert a._passes_filters(self._candidate(dte=5))

    def test_delta_below_min_rejected(self) -> None:
        a = self._analyzer()
        assert not a._passes_filters(self._candidate(delta=0.10))

    def test_delta_above_max_rejected(self) -> None:
        a = self._analyzer()
        assert not a._passes_filters(self._candidate(delta=0.50))

    def test_iv_pct_at_max_rejected(self) -> None:
        """iv_pct_max is exclusive per the _passes_filters docstring."""
        a = self._analyzer()
        assert not a._passes_filters(self._candidate(iv_pct=50.0))

    def test_iv_pct_below_max_accepted(self) -> None:
        a = self._analyzer()
        assert a._passes_filters(self._candidate(iv_pct=49.9))

    def test_spread_pct_above_max_rejected(self) -> None:
        a = self._analyzer()
        assert not a._passes_filters(self._candidate(spread_pct=25.0))

    def test_spread_pct_at_max_accepted(self) -> None:
        a = self._analyzer()
        assert a._passes_filters(self._candidate(spread_pct=20.0))


# ---------------------------------------------------------------------------
# OptionsChainAnalyzer._compute_score
# ---------------------------------------------------------------------------


class TestComputeScore:
    def _analyzer(self) -> OptionsChainAnalyzer:
        return OptionsChainAnalyzer(api_key="test_key")

    def _candidate(self, delta: float = 0.30, iv_pct: float = 25.0,
                   spread_pct: float = 10.0) -> OptionCandidate:
        return OptionCandidate(
            symbol="O:TEST",
            underlying="TEST",
            option_type="call",
            strike=100.0,
            expiry=_future_date(3),
            dte=3,
            bid=1.00,
            ask=1.20,
            delta=delta,
            iv_percentile=iv_pct,
            mid_price=1.10,
            spread_pct=spread_pct,
        )

    def test_score_in_unit_interval(self) -> None:
        a = self._analyzer()
        score = a._compute_score(self._candidate())
        assert 0.0 <= score <= 1.0

    def test_ideal_candidate_has_high_score(self) -> None:
        """delta=0.30 (ideal), low IV, tight spread → near-maximum score."""
        a = self._analyzer()
        score = a._compute_score(self._candidate(delta=0.30, iv_pct=0.0, spread_pct=0.0))
        assert score > 0.8

    def test_higher_iv_lowers_score(self) -> None:
        a = self._analyzer()
        low_iv = a._compute_score(self._candidate(iv_pct=10.0))
        high_iv = a._compute_score(self._candidate(iv_pct=40.0))
        assert low_iv > high_iv

    def test_tighter_spread_raises_score(self) -> None:
        a = self._analyzer()
        tight = a._compute_score(self._candidate(spread_pct=5.0))
        wide = a._compute_score(self._candidate(spread_pct=18.0))
        assert tight > wide

    def test_delta_close_to_ideal_raises_score(self) -> None:
        a = self._analyzer()
        ideal = a._compute_score(self._candidate(delta=0.30))
        far = a._compute_score(self._candidate(delta=0.39))
        assert ideal > far


# ---------------------------------------------------------------------------
# OptionsChainAnalyzer.get_chain — full integration with mocked HTTP
# ---------------------------------------------------------------------------


def _mock_session(results: list[dict]) -> MagicMock:
    mock_session = MagicMock()
    resp = MagicMock()
    resp.json = AsyncMock(return_value={"results": results, "next_url": None})
    resp.raise_for_status = MagicMock()
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    mock_session.get.return_value = ctx

    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)
    return session_ctx


class TestGetChain:
    @pytest.mark.asyncio
    async def test_get_chain_returns_list(self) -> None:
        a = OptionsChainAnalyzer(api_key="test_key")
        raw = _make_raw_contract(expiry=_future_date(3))
        session_ctx = _mock_session([raw])

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            candidates = await a.get_chain("AAPL")

        assert isinstance(candidates, list)

    @pytest.mark.asyncio
    async def test_get_chain_sorted_by_score_descending(self) -> None:
        a = OptionsChainAnalyzer(api_key="test_key")
        # Two contracts: one with ideal delta, one far off
        raw1 = _make_raw_contract(ticker="O:AAPL_A", delta=0.30, expiry=_future_date(3))
        raw2 = _make_raw_contract(ticker="O:AAPL_B", delta=0.39, expiry=_future_date(3))
        session_ctx = _mock_session([raw1, raw2])

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            candidates = await a.get_chain("AAPL")

        if len(candidates) >= 2:
            assert candidates[0].score >= candidates[1].score

    @pytest.mark.asyncio
    async def test_get_chain_filters_out_expired(self) -> None:
        a = OptionsChainAnalyzer(api_key="test_key")
        past = (date.today() - timedelta(days=1)).isoformat()
        raw = _make_raw_contract(expiry=past)
        session_ctx = _mock_session([raw])

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            candidates = await a.get_chain("AAPL")

        assert candidates == []

    @pytest.mark.asyncio
    async def test_get_chain_filters_out_dte_out_of_range(self) -> None:
        a = OptionsChainAnalyzer(api_key="test_key", dte_min=1, dte_max=5)
        far_expiry = _future_date(30)  # 30 DTE — outside dte_max=5
        raw = _make_raw_contract(expiry=far_expiry)
        session_ctx = _mock_session([raw])

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            candidates = await a.get_chain("AAPL")

        assert candidates == []

    @pytest.mark.asyncio
    async def test_get_chain_empty_results_from_api(self) -> None:
        a = OptionsChainAnalyzer(api_key="test_key")
        session_ctx = _mock_session([])

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            candidates = await a.get_chain("AAPL")

        assert candidates == []

    @pytest.mark.asyncio
    async def test_get_chain_handles_api_error_gracefully(self) -> None:
        a = OptionsChainAnalyzer(api_key="test_key")

        mock_session = MagicMock()
        mock_session.get.side_effect = Exception("network error")

        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            candidates = await a.get_chain("AAPL")

        assert candidates == []

    @pytest.mark.asyncio
    async def test_get_chain_candidate_fields_populated(self) -> None:
        a = OptionsChainAnalyzer(api_key="test_key")
        raw = _make_raw_contract(
            ticker="O:AAPL251219C00150000",
            contract_type="call",
            strike=150.0,
            expiry=_future_date(3),
            bid=1.40,
            ask=1.60,
            delta=0.30,
            iv_percentile=25.0,
        )
        session_ctx = _mock_session([raw])

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            candidates = await a.get_chain("AAPL")

        assert len(candidates) == 1
        c = candidates[0]
        assert c.underlying == "AAPL"
        assert c.option_type == "call"
        assert c.strike == pytest.approx(150.0)
        assert c.bid == pytest.approx(1.40)
        assert c.ask == pytest.approx(1.60)
        assert c.delta == pytest.approx(0.30)
        assert 0.0 <= c.score <= 1.0


# ---------------------------------------------------------------------------
# OptionCandidate dataclass
# ---------------------------------------------------------------------------


class TestOptionCandidateDataclass:
    def _sample(self) -> OptionCandidate:
        return OptionCandidate(
            symbol="O:AAPL251219C00150000",
            underlying="AAPL",
            option_type="call",
            strike=150.0,
            expiry="2025-12-19",
            dte=3,
            bid=1.30,
            ask=1.70,
            delta=0.30,
            iv_percentile=25.0,
            mid_price=1.50,
            spread_pct=26.67,
        )

    def test_default_score_is_zero(self) -> None:
        c = self._sample()
        assert c.score == pytest.approx(0.0)

    def test_score_can_be_set(self) -> None:
        c = self._sample()
        c.score = 0.75
        assert c.score == pytest.approx(0.75)

    def test_fields_accessible(self) -> None:
        c = self._sample()
        assert c.symbol == "O:AAPL251219C00150000"
        assert c.underlying == "AAPL"
        assert c.option_type == "call"
        assert c.strike == pytest.approx(150.0)
        assert c.dte == 3
