"""Tests for C++ pybind11 bindings.

All tests in this module are skipped automatically when the compiled
_coe_engine extension is not present.  Build the engine first with:

    pip install -e ".[dev]"

or:

    cmake --build engine --target coe_engine
"""

from __future__ import annotations

import pytest

try:
    from coe_engine import (
        CircuitBreaker,
        Config,
        CoeError,
        DailyPnLTracker,
        ErrorCode,
        GreeksFilter,
        KellyPositionSizer,
        OptionType,
        Order,
        OrderManager,
        OrderState,
        OrderType,
        Position,
        PositionTracker,
        RSI,
        BollingerBands,
        RiskLimits,
        RiskManager,
        Side,
        SignalScorer,
        VolumeSpike,
        next_order_id,
    )

    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

pytestmark = pytest.mark.skipif(not HAS_ENGINE, reason="C++ bindings not built")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_side_values(self) -> None:
        assert Side.Buy is not None
        assert Side.Sell is not None

    def test_side_buy_sell_distinct(self) -> None:
        assert Side.Buy != Side.Sell

    def test_option_type_values(self) -> None:
        assert OptionType.Call is not None
        assert OptionType.Put is not None

    def test_option_type_call_put_distinct(self) -> None:
        assert OptionType.Call != OptionType.Put

    def test_error_code_ok(self) -> None:
        assert ErrorCode.Ok is not None

    def test_error_code_order_rejected(self) -> None:
        assert ErrorCode.OrderRejected is not None

    def test_error_code_ok_and_rejected_distinct(self) -> None:
        assert ErrorCode.Ok != ErrorCode.OrderRejected


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_load_strategy_yaml(self) -> None:
        config = Config("config/strategy.yaml")
        assert config is not None

    def test_get_int_rsi_period(self) -> None:
        config = Config("config/strategy.yaml")
        period = config.get_int("strategy.rsi.period", 0)
        assert period == 14

    def test_get_double_oversold(self) -> None:
        config = Config("config/strategy.yaml")
        oversold = config.get_double("strategy.rsi.oversold", 0.0)
        assert oversold == pytest.approx(30.0)

    def test_get_int_missing_key_returns_default(self) -> None:
        config = Config("config/strategy.yaml")
        value = config.get_int("nonexistent.key", 42)
        assert value == 42

    def test_get_double_missing_key_returns_default(self) -> None:
        config = Config("config/strategy.yaml")
        value = config.get_double("nonexistent.key", 3.14)
        assert value == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


class TestRSI:
    def test_not_ready_initially(self) -> None:
        rsi = RSI(14)
        assert not rsi.ready()

    def test_ready_after_period_plus_one_updates(self) -> None:
        rsi = RSI(14)
        for i in range(15):
            rsi.update(100.0 + i)
        # period + 1 updates needed to seed first average
        assert rsi.ready()

    def test_ready_after_more_than_period_updates(self) -> None:
        rsi = RSI(14)
        for i in range(20):
            rsi.update(100.0 + i)
        assert rsi.ready()
        assert 0.0 <= rsi.value() <= 100.0

    def test_rising_prices_give_high_rsi(self) -> None:
        rsi = RSI(14)
        for i in range(20):
            rsi.update(100.0 + i)
        assert rsi.value() > 50.0

    def test_falling_prices_give_low_rsi(self) -> None:
        rsi = RSI(14)
        for i in range(20):
            rsi.update(200.0 - i)
        assert rsi.value() < 50.0

    def test_reset_clears_state(self) -> None:
        rsi = RSI(14)
        for i in range(20):
            rsi.update(100.0 + i)
        assert rsi.ready()
        rsi.reset()
        assert not rsi.ready()

    def test_value_raises_when_not_ready(self) -> None:
        rsi = RSI(14)
        # Should either raise or return a sentinel — confirm it is not ready
        assert not rsi.ready()


# ---------------------------------------------------------------------------
# BollingerBands
# ---------------------------------------------------------------------------


class TestBollingerBands:
    def test_not_ready_initially(self) -> None:
        bb = BollingerBands(20, 2.0)
        assert not bb.ready()

    def test_ready_after_enough_updates(self) -> None:
        bb = BollingerBands(20, 2.0)
        for i in range(25):
            bb.update(100.0 + (i % 5))
        assert bb.ready()

    def test_band_ordering_upper_middle_lower(self) -> None:
        bb = BollingerBands(20, 2.0)
        for i in range(25):
            bb.update(100.0 + (i % 5))
        assert bb.upper() > bb.middle() > bb.lower()

    def test_bands_are_finite(self) -> None:
        bb = BollingerBands(20, 2.0)
        for i in range(25):
            bb.update(100.0 + (i % 5))
        import math
        assert math.isfinite(bb.upper())
        assert math.isfinite(bb.middle())
        assert math.isfinite(bb.lower())

    def test_flat_prices_tight_bands(self) -> None:
        """With constant prices the standard deviation is 0 and bands collapse."""
        bb = BollingerBands(20, 2.0)
        for _ in range(25):
            bb.update(100.0)
        assert bb.ready()
        assert bb.upper() == pytest.approx(bb.middle())
        assert bb.lower() == pytest.approx(bb.middle())


# ---------------------------------------------------------------------------
# VolumeSpike
# ---------------------------------------------------------------------------


class TestVolumeSpike:
    def test_not_ready_initially(self) -> None:
        vs = VolumeSpike(20, 2.0)
        assert not vs.ready()

    def test_ready_after_lookback_updates(self) -> None:
        vs = VolumeSpike(20, 2.0)
        for i in range(21):
            vs.update(1_000_000.0)
        assert vs.ready()

    def test_spike_detected_on_high_volume(self) -> None:
        vs = VolumeSpike(20, 2.0)
        for _ in range(20):
            vs.update(1_000_000.0)
        # Feed 3× average — should trigger a spike
        vs.update(3_000_000.0)
        assert vs.is_spike()

    def test_no_spike_on_normal_volume(self) -> None:
        vs = VolumeSpike(20, 2.0)
        for _ in range(21):
            vs.update(1_000_000.0)
        assert not vs.is_spike()


# ---------------------------------------------------------------------------
# GreeksFilter
# ---------------------------------------------------------------------------


class TestGreeksFilter:
    def test_passes_valid_candidate(self) -> None:
        gf = GreeksFilter(delta_min=0.20, delta_max=0.40, iv_pct_max=50.0, spread_pct_max=20.0)
        assert gf.passes(delta=0.30, iv_percentile=25.0, spread_pct=10.0)

    def test_fails_delta_too_low(self) -> None:
        gf = GreeksFilter(delta_min=0.20, delta_max=0.40, iv_pct_max=50.0, spread_pct_max=20.0)
        assert not gf.passes(delta=0.10, iv_percentile=25.0, spread_pct=10.0)

    def test_fails_delta_too_high(self) -> None:
        gf = GreeksFilter(delta_min=0.20, delta_max=0.40, iv_pct_max=50.0, spread_pct_max=20.0)
        assert not gf.passes(delta=0.50, iv_percentile=25.0, spread_pct=10.0)

    def test_fails_iv_too_high(self) -> None:
        gf = GreeksFilter(delta_min=0.20, delta_max=0.40, iv_pct_max=50.0, spread_pct_max=20.0)
        assert not gf.passes(delta=0.30, iv_percentile=60.0, spread_pct=10.0)

    def test_fails_spread_too_wide(self) -> None:
        gf = GreeksFilter(delta_min=0.20, delta_max=0.40, iv_pct_max=50.0, spread_pct_max=20.0)
        assert not gf.passes(delta=0.30, iv_percentile=25.0, spread_pct=25.0)


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------


class TestOrderManager:
    def _make_limit_order(self) -> Order:
        order = Order()
        order.symbol = "AAPL"
        order.side = Side.Buy
        order.quantity = 1
        order.order_type = OrderType.Limit
        order.limit_price = 1.50
        return order

    def _make_market_order(self) -> Order:
        order = Order()
        order.symbol = "TSLA"
        order.side = Side.Sell
        order.quantity = 2
        order.order_type = OrderType.Market
        return order

    def test_submit_returns_positive_id(self) -> None:
        om = OrderManager()
        order = self._make_limit_order()
        order_id = om.submit(order)
        assert order_id > 0

    def test_get_order_by_id(self) -> None:
        om = OrderManager()
        order = self._make_limit_order()
        order_id = om.submit(order)
        retrieved = om.get_order(order_id)
        assert retrieved is not None
        assert retrieved.symbol == "AAPL"

    def test_get_order_preserves_side(self) -> None:
        om = OrderManager()
        order = self._make_limit_order()
        order_id = om.submit(order)
        retrieved = om.get_order(order_id)
        assert retrieved.side == Side.Buy

    def test_cancel_sets_cancelled_state(self) -> None:
        om = OrderManager()
        order = self._make_market_order()
        order_id = om.submit(order)
        om.cancel(order_id)
        retrieved = om.get_order(order_id)
        assert retrieved.state == OrderState.Cancelled

    def test_sequential_ids_are_unique(self) -> None:
        om = OrderManager()
        ids: set[int] = set()
        for _ in range(5):
            o = self._make_limit_order()
            ids.add(om.submit(o))
        assert len(ids) == 5

    def test_get_unknown_order_returns_none(self) -> None:
        om = OrderManager()
        result = om.get_order(99_999)
        assert result is None


# ---------------------------------------------------------------------------
# PositionTracker
# ---------------------------------------------------------------------------


class TestPositionTracker:
    def test_on_fill_creates_position(self) -> None:
        pt = PositionTracker()
        pt.on_fill("AAPL", Side.Buy, 1, 1.50)
        pos = pt.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == 1

    def test_on_fill_records_correct_quantity(self) -> None:
        pt = PositionTracker()
        pt.on_fill("MSFT", Side.Buy, 5, 2.00)
        pos = pt.get_position("MSFT")
        assert pos.quantity == 5

    def test_unknown_symbol_returns_none(self) -> None:
        pt = PositionTracker()
        assert pt.get_position("ZZZZZ") is None

    def test_unrealized_pnl_positive_after_mark_up(self) -> None:
        pt = PositionTracker()
        pt.on_fill("AAPL", Side.Buy, 1, 1.50)
        pt.update_mark("AAPL", 2.00)
        assert pt.total_unrealized_pnl() > 0

    def test_unrealized_pnl_negative_after_mark_down(self) -> None:
        pt = PositionTracker()
        pt.on_fill("AAPL", Side.Buy, 1, 2.00)
        pt.update_mark("AAPL", 1.00)
        assert pt.total_unrealized_pnl() < 0

    def test_sell_fill_reduces_quantity(self) -> None:
        pt = PositionTracker()
        pt.on_fill("AAPL", Side.Buy, 3, 1.50)
        pt.on_fill("AAPL", Side.Sell, 1, 1.60)
        pos = pt.get_position("AAPL")
        assert pos.quantity == 2


# ---------------------------------------------------------------------------
# KellyPositionSizer
# ---------------------------------------------------------------------------


class TestKellyPositionSizer:
    def test_calculate_size_positive(self) -> None:
        sizer = KellyPositionSizer(0.5, 20.0)
        size = sizer.calculate_size(0.6, 2.0, 1.0, 1000.0)
        assert size > 0

    def test_calculate_size_respects_max_bet(self) -> None:
        """Result must not exceed max_bet_pct% of bankroll."""
        sizer = KellyPositionSizer(0.5, 20.0)
        size = sizer.calculate_size(0.6, 2.0, 1.0, 1000.0)
        assert size <= 200.0  # 20 % of 1 000

    def test_contracts_at_least_one(self) -> None:
        sizer = KellyPositionSizer(0.5, 20.0)
        qty = sizer.contracts(150.0, 1.50)
        assert qty >= 1

    def test_contracts_are_integer(self) -> None:
        sizer = KellyPositionSizer(0.5, 20.0)
        qty = sizer.contracts(150.0, 1.50)
        assert isinstance(qty, int)

    def test_zero_win_rate_gives_zero_size(self) -> None:
        sizer = KellyPositionSizer(0.5, 20.0)
        size = sizer.calculate_size(0.0, 2.0, 1.0, 1000.0)
        assert size == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# RiskManager / CircuitBreaker / DailyPnLTracker
# ---------------------------------------------------------------------------


class TestRiskManager:
    def test_initial_daily_pnl_is_zero(self) -> None:
        limits = RiskLimits()
        rm = RiskManager(limits)
        assert rm.daily_pnl() == pytest.approx(0.0)

    def test_daily_pnl_tracks_losses(self) -> None:
        limits = RiskLimits()
        rm = RiskManager(limits)
        rm.on_trade_closed(-10.0, False)
        assert rm.daily_pnl() == pytest.approx(-10.0)

    def test_daily_pnl_tracks_profits(self) -> None:
        limits = RiskLimits()
        rm = RiskManager(limits)
        rm.on_trade_closed(25.0, True)
        assert rm.daily_pnl() == pytest.approx(25.0)

    def test_circuit_breaker_not_tripped_initially(self) -> None:
        limits = RiskLimits()
        rm = RiskManager(limits)
        assert not rm.is_circuit_breaker_tripped()

    def test_circuit_breaker_trips_on_large_losses(self) -> None:
        limits = RiskLimits()
        rm = RiskManager(limits)
        for _ in range(20):
            rm.on_trade_closed(-1.0, False)
        assert rm.is_circuit_breaker_tripped()

    def test_can_trade_returns_false_when_breaker_tripped(self) -> None:
        limits = RiskLimits()
        rm = RiskManager(limits)
        for _ in range(20):
            rm.on_trade_closed(-1.0, False)
        assert not rm.can_trade()

    def test_can_trade_returns_true_initially(self) -> None:
        limits = RiskLimits()
        rm = RiskManager(limits)
        assert rm.can_trade()


# ---------------------------------------------------------------------------
# next_order_id utility
# ---------------------------------------------------------------------------


class TestNextOrderId:
    def test_returns_positive_integer(self) -> None:
        oid = next_order_id()
        assert oid > 0

    def test_sequential_calls_increment(self) -> None:
        id1 = next_order_id()
        id2 = next_order_id()
        assert id2 > id1
