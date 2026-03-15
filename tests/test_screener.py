"""Tests for OptionsScreener and the module-level _compute_rsi helper.

All external HTTP calls are patched with aiohttp mocks so no live network
access is required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signals.screener import OptionsScreener, _compute_rsi


# ---------------------------------------------------------------------------
# _compute_rsi — pure-Python helper (no network, no mocks needed)
# ---------------------------------------------------------------------------


class TestComputeRsi:
    def test_returns_none_when_insufficient_bars(self) -> None:
        # period=14 requires at least 15 prices
        assert _compute_rsi([100.0] * 10, period=14) is None

    def test_returns_none_on_exact_minimum_minus_one(self) -> None:
        assert _compute_rsi([100.0] * 14, period=14) is None

    def test_returns_value_at_minimum_bars(self) -> None:
        prices = [100.0 + i for i in range(15)]
        result = _compute_rsi(prices, period=14)
        assert result is not None
        assert 0.0 <= result <= 100.0

    def test_monotonically_rising_prices_give_rsi_above_50(self) -> None:
        prices = [100.0 + i for i in range(20)]
        result = _compute_rsi(prices, period=14)
        assert result is not None
        assert result > 50.0

    def test_monotonically_falling_prices_give_rsi_below_50(self) -> None:
        prices = [200.0 - i for i in range(20)]
        result = _compute_rsi(prices, period=14)
        assert result is not None
        assert result < 50.0

    def test_all_gains_returns_100(self) -> None:
        """When avg_loss == 0 the function should return 100.0."""
        prices = [100.0 + i for i in range(30)]
        result = _compute_rsi(prices, period=14)
        assert result == pytest.approx(100.0)

    def test_all_losses_returns_near_zero(self) -> None:
        prices = [100.0 - i * 0.5 for i in range(30)]
        result = _compute_rsi(prices, period=14)
        assert result is not None
        assert result < 5.0

    def test_result_bounded_between_0_and_100(self) -> None:
        import random

        random.seed(42)
        prices = [100.0 + random.gauss(0, 5) for _ in range(50)]
        result = _compute_rsi(prices, period=14)
        assert result is not None
        assert 0.0 <= result <= 100.0

    def test_custom_period(self) -> None:
        prices = [100.0 + i for i in range(15)]
        result_7 = _compute_rsi(prices, period=7)
        result_14 = _compute_rsi(prices, period=7)
        # Both should produce valid results for a shorter period
        assert result_7 is not None
        assert result_14 is not None


# ---------------------------------------------------------------------------
# OptionsScreener — constructor
# ---------------------------------------------------------------------------


class TestOptionsScreenerInit:
    def test_requires_api_key_or_env_var(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            # Remove POLYGON_API_KEY from the environment if present
            import os
            os.environ.pop("POLYGON_API_KEY", None)
            with pytest.raises(ValueError, match="Polygon API key"):
                OptionsScreener(api_key=None)

    def test_accepts_explicit_api_key(self) -> None:
        screener = OptionsScreener(api_key="test_key")
        assert screener is not None

    def test_accepts_env_var(self) -> None:
        with patch.dict("os.environ", {"POLYGON_API_KEY": "env_key"}):
            screener = OptionsScreener()
            assert screener is not None

    def test_default_parameter_values(self) -> None:
        screener = OptionsScreener(api_key="test_key")
        assert screener.min_market_cap == 1e9
        assert screener.min_volume == 1_000_000
        assert screener.rsi_threshold == 35.0
        assert screener.rsi_period == 14
        assert screener.bars_lookback == 20

    def test_custom_parameter_values(self) -> None:
        screener = OptionsScreener(
            api_key="test_key",
            min_market_cap=5e9,
            min_volume=500_000,
            rsi_threshold=30.0,
            rsi_period=21,
            bars_lookback=30,
        )
        assert screener.min_market_cap == 5e9
        assert screener.min_volume == 500_000
        assert screener.rsi_threshold == 30.0
        assert screener.rsi_period == 21
        assert screener.bars_lookback == 30


# ---------------------------------------------------------------------------
# Helpers — build mock aiohttp session
# ---------------------------------------------------------------------------


def _mock_session_for_get(json_responses: list) -> MagicMock:
    """Return a MagicMock aiohttp.ClientSession whose get() yields json_responses
    in order. Each entry in json_responses is the dict/list returned by resp.json().
    """
    mock_session = MagicMock()

    call_count = 0

    async def _fake_json():
        nonlocal call_count
        response = json_responses[min(call_count, len(json_responses) - 1)]
        call_count += 1
        return response

    def _make_resp(status: int = 200) -> MagicMock:
        resp = MagicMock()
        resp.status = status
        resp.json = _fake_json
        resp.raise_for_status = MagicMock()  # no-op for 200
        return resp

    # aiohttp context manager protocol: `async with session.get(...) as resp`
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=_make_resp())
    ctx.__aexit__ = AsyncMock(return_value=False)
    mock_session.get.return_value = ctx

    return mock_session


# ---------------------------------------------------------------------------
# OptionsScreener._compute_rsi (instance delegation)
# ---------------------------------------------------------------------------


class TestScreenerComputeRsi:
    """The screener delegates to the module-level _compute_rsi.
    Test via the public scan() flow indirectly and via the helper directly.
    """

    def test_rising_prices(self) -> None:
        prices = [100.0 + i for i in range(20)]
        assert _compute_rsi(prices, 14) > 50.0  # type: ignore[operator]

    def test_falling_prices(self) -> None:
        prices = [200.0 - i for i in range(20)]
        assert _compute_rsi(prices, 14) < 50.0  # type: ignore[operator]


# ---------------------------------------------------------------------------
# OptionsScreener.scan — volume / market-cap filter
# ---------------------------------------------------------------------------


class TestScanVolumeFilter:
    @pytest.mark.asyncio
    async def test_low_volume_ticker_excluded(self) -> None:
        """A ticker with volume below min_volume must not appear in candidates."""
        # snapshot endpoint returns one ticker that passes cap filter but fails volume
        snapshot_data = {
            "tickers": [
                {
                    "ticker": "LOW_VOL",
                    "market_cap": 5e9,
                    "day": {"v": 100, "c": 10.0},  # volume << 1_000_000
                    "prevDay": {"c": 10.5},
                }
            ]
        }

        screener = OptionsScreener(api_key="test_key")

        with patch("aiohttp.ClientSession") as MockSession:
            session_instance = MagicMock()
            MockSession.return_value.__aenter__ = AsyncMock(return_value=session_instance)
            MockSession.return_value.__aexit__ = AsyncMock(return_value=False)

            # Snapshot response
            resp_snapshot = MagicMock()
            resp_snapshot.json = AsyncMock(return_value=snapshot_data)
            resp_snapshot.raise_for_status = MagicMock()

            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=resp_snapshot)
            ctx.__aexit__ = AsyncMock(return_value=False)
            session_instance.get.return_value = ctx

            candidates = await screener.scan()

        assert not any(c["symbol"] == "LOW_VOL" for c in candidates)

    @pytest.mark.asyncio
    async def test_low_market_cap_ticker_excluded(self) -> None:
        """A ticker below min_market_cap must not appear in candidates."""
        snapshot_data = {
            "tickers": [
                {
                    "ticker": "TINY",
                    "market_cap": 1e6,  # $1 M — below $1 B threshold
                    "day": {"v": 5_000_000, "c": 1.0},
                    "prevDay": {"c": 1.1},
                }
            ]
        }

        screener = OptionsScreener(api_key="test_key")

        with patch("aiohttp.ClientSession") as MockSession:
            session_instance = MagicMock()
            MockSession.return_value.__aenter__ = AsyncMock(return_value=session_instance)
            MockSession.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = MagicMock()
            resp.json = AsyncMock(return_value=snapshot_data)
            resp.raise_for_status = MagicMock()
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=resp)
            ctx.__aexit__ = AsyncMock(return_value=False)
            session_instance.get.return_value = ctx

            candidates = await screener.scan()

        assert not any(c["symbol"] == "TINY" for c in candidates)


# ---------------------------------------------------------------------------
# OptionsScreener.scan — RSI filter
# ---------------------------------------------------------------------------


class TestScanRsiFilter:
    @pytest.mark.asyncio
    async def test_oversold_ticker_included(self) -> None:
        """A ticker whose bars produce RSI < rsi_threshold appears in candidates."""
        # Produce 25 strictly-falling closes — will yield very low RSI
        falling_closes = [200.0 - i for i in range(25)]
        bars_data = {"results": [{"c": p} for p in falling_closes]}

        snapshot_data = {
            "tickers": [
                {
                    "ticker": "OVER_SOLD",
                    "market_cap": 50e9,
                    "day": {"v": 10_000_000, "c": falling_closes[-1]},
                    "prevDay": {"c": falling_closes[-2]},
                    "lastTrade": {"p": falling_closes[-1]},
                }
            ]
        }

        screener = OptionsScreener(api_key="test_key", rsi_threshold=35.0)

        call_index = 0
        responses = [snapshot_data, bars_data]

        async def _fake_json() -> dict:
            nonlocal call_index
            data = responses[min(call_index, len(responses) - 1)]
            call_index += 1
            return data

        with patch("aiohttp.ClientSession") as MockSession:
            session_instance = MagicMock()
            MockSession.return_value.__aenter__ = AsyncMock(return_value=session_instance)
            MockSession.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = MagicMock()
            resp.json = _fake_json
            resp.raise_for_status = MagicMock()
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=resp)
            ctx.__aexit__ = AsyncMock(return_value=False)
            session_instance.get.return_value = ctx

            candidates = await screener.scan()

        symbols = [c["symbol"] for c in candidates]
        assert "OVER_SOLD" in symbols

    @pytest.mark.asyncio
    async def test_candidate_dict_has_required_keys(self) -> None:
        """Each returned candidate must carry symbol, price, volume, rsi, market_cap."""
        falling_closes = [200.0 - i for i in range(25)]
        bars_data = {"results": [{"c": p} for p in falling_closes]}
        snapshot_data = {
            "tickers": [
                {
                    "ticker": "CAND",
                    "market_cap": 50e9,
                    "day": {"v": 10_000_000, "c": falling_closes[-1]},
                    "prevDay": {"c": falling_closes[-2]},
                    "lastTrade": {"p": falling_closes[-1]},
                }
            ]
        }

        screener = OptionsScreener(api_key="test_key", rsi_threshold=35.0)
        call_index = 0
        responses = [snapshot_data, bars_data]

        async def _fake_json() -> dict:
            nonlocal call_index
            data = responses[min(call_index, len(responses) - 1)]
            call_index += 1
            return data

        with patch("aiohttp.ClientSession") as MockSession:
            session_instance = MagicMock()
            MockSession.return_value.__aenter__ = AsyncMock(return_value=session_instance)
            MockSession.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = MagicMock()
            resp.json = _fake_json
            resp.raise_for_status = MagicMock()
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=resp)
            ctx.__aexit__ = AsyncMock(return_value=False)
            session_instance.get.return_value = ctx

            candidates = await screener.scan()

        for c in candidates:
            assert "symbol" in c
            assert "price" in c
            assert "volume" in c
            assert "rsi" in c
            assert "market_cap" in c


# ---------------------------------------------------------------------------
# OptionsScreener.scan — error handling
# ---------------------------------------------------------------------------


class TestScanErrorHandling:
    @pytest.mark.asyncio
    async def test_snapshot_fetch_failure_returns_empty_list(self) -> None:
        """When the snapshot endpoint raises, scan() returns an empty list."""
        screener = OptionsScreener(api_key="test_key")

        with patch("aiohttp.ClientSession") as MockSession:
            session_instance = MagicMock()
            MockSession.return_value.__aenter__ = AsyncMock(return_value=session_instance)
            MockSession.return_value.__aexit__ = AsyncMock(return_value=False)

            # Raise on get()
            session_instance.get.side_effect = Exception("network error")

            candidates = await screener.scan()

        assert candidates == []

    @pytest.mark.asyncio
    async def test_insufficient_bars_ticker_skipped(self) -> None:
        """Tickers with fewer bars than rsi_period + 1 are silently skipped."""
        bars_data = {"results": [{"c": 100.0}] * 5}  # only 5 bars — far too few
        snapshot_data = {
            "tickers": [
                {
                    "ticker": "FEW_BARS",
                    "market_cap": 50e9,
                    "day": {"v": 5_000_000, "c": 100.0},
                    "prevDay": {"c": 101.0},
                }
            ]
        }

        screener = OptionsScreener(api_key="test_key")
        call_index = 0
        responses = [snapshot_data, bars_data]

        async def _fake_json() -> dict:
            nonlocal call_index
            data = responses[min(call_index, len(responses) - 1)]
            call_index += 1
            return data

        with patch("aiohttp.ClientSession") as MockSession:
            session_instance = MagicMock()
            MockSession.return_value.__aenter__ = AsyncMock(return_value=session_instance)
            MockSession.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = MagicMock()
            resp.json = _fake_json
            resp.raise_for_status = MagicMock()
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=resp)
            ctx.__aexit__ = AsyncMock(return_value=False)
            session_instance.get.return_value = ctx

            candidates = await screener.scan()

        assert not any(c["symbol"] == "FEW_BARS" for c in candidates)


# ---------------------------------------------------------------------------
# OptionsScreener.get_bars
# ---------------------------------------------------------------------------


class TestGetBars:
    @pytest.mark.asyncio
    async def test_get_bars_returns_list(self) -> None:
        bars = [{"t": 1000 + i, "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 500_000}
                for i in range(20)]
        bars_data = {"results": bars}

        screener = OptionsScreener(api_key="test_key")

        with patch("aiohttp.ClientSession") as MockSession:
            session_instance = MagicMock()
            MockSession.return_value.__aenter__ = AsyncMock(return_value=session_instance)
            MockSession.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = MagicMock()
            resp.json = AsyncMock(return_value=bars_data)
            resp.raise_for_status = MagicMock()
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=resp)
            ctx.__aexit__ = AsyncMock(return_value=False)
            session_instance.get.return_value = ctx

            result = await screener.get_bars("AAPL", days=20)

        assert isinstance(result, list)
        assert len(result) == 20

    @pytest.mark.asyncio
    async def test_get_bars_empty_when_no_results(self) -> None:
        screener = OptionsScreener(api_key="test_key")

        with patch("aiohttp.ClientSession") as MockSession:
            session_instance = MagicMock()
            MockSession.return_value.__aenter__ = AsyncMock(return_value=session_instance)
            MockSession.return_value.__aexit__ = AsyncMock(return_value=False)

            resp = MagicMock()
            resp.json = AsyncMock(return_value={})  # no "results" key
            resp.raise_for_status = MagicMock()
            ctx = MagicMock()
            ctx.__aenter__ = AsyncMock(return_value=resp)
            ctx.__aexit__ = AsyncMock(return_value=False)
            session_instance.get.return_value = ctx

            result = await screener.get_bars("AAPL", days=20)

        assert result == []
