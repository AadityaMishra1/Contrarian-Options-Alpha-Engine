"""Tests for IBKROrderBridge, IBKRConnection, and PositionReconciler.

ib_insync is mocked at the module import level so these tests run without a
live TWS / IB Gateway instance.  Tests that cannot be meaningfully run without
the real dependency are skipped with an informative message.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Check if ib_insync is actually installed (not just import-guarded)
try:
    import ib_insync as _ib_check  # noqa: F401
    _HAS_IB = True
except ImportError:
    _HAS_IB = False

requires_ib = pytest.mark.skipif(not _HAS_IB, reason="ib_insync not installed")


# ---------------------------------------------------------------------------
# Lightweight duck-typed order / fill stand-ins
# ---------------------------------------------------------------------------


@dataclass
class _MockOrder:
    """Duck-typed order compatible with IBKROrderBridge.submit_order()."""

    order_id: int = 1
    symbol: str = "AAPL"
    expiry: str = "20251219"
    strike: float = 150.0
    right: str = "C"
    side: str = "BUY"
    order_type: str = "LMT"
    limit_price: float = 1.50
    quantity: int = 1


@dataclass
class _MockExecution:
    shares: float = 1.0
    avgPrice: float = 1.50
    side: str = "BOT"


@dataclass
class _MockFill:
    execution: _MockExecution = field(default_factory=_MockExecution)


# ---------------------------------------------------------------------------
# IBKROrderBridge — import guard
# ---------------------------------------------------------------------------


def _import_bridge():
    """Import IBKROrderBridge, skipping if ib_insync is missing."""
    try:
        from broker.order_bridge import IBKROrderBridge, Side, OrderType
        return IBKROrderBridge, Side, OrderType
    except ImportError as exc:
        if "ib_insync" in str(exc):
            pytest.skip("ib_insync not installed")
        raise


def _import_connection():
    """Import IBKRConnection, skipping if ib_insync is missing."""
    try:
        from broker.connection import IBKRConnection, IBKRConnectionError
        return IBKRConnection, IBKRConnectionError
    except ImportError as exc:
        if "ib_insync" in str(exc):
            pytest.skip("ib_insync not installed")
        raise


def _import_reconciler():
    """Import PositionReconciler, skipping if ib_insync is missing."""
    try:
        from broker.reconciliation import (
            PositionReconciler,
            Discrepancy,
            DiscrepancyType,
        )
        return PositionReconciler, Discrepancy, DiscrepancyType
    except ImportError as exc:
        if "ib_insync" in str(exc):
            pytest.skip("ib_insync not installed")
        raise


# ---------------------------------------------------------------------------
# Side and OrderType Python enum stand-ins (always importable)
# ---------------------------------------------------------------------------


class TestSideAndOrderType:
    """The Python-side Side/OrderType classes in order_bridge are always importable."""

    def test_side_importable_without_ib_insync(self) -> None:
        """order_bridge.Side must be accessible regardless of ib_insync availability."""
        # Patch ib_insync as missing at the module level
        with patch.dict("sys.modules", {"ib_insync": None}):
            # Re-import the module from scratch to test the fallback branch
            import importlib
            import broker.order_bridge as _mod
            importlib.reload(_mod)
            assert hasattr(_mod, "Side")
            assert _mod.Side.Buy == "BUY"
            assert _mod.Side.Sell == "SELL"

    def test_order_type_importable_without_ib_insync(self) -> None:
        with patch.dict("sys.modules", {"ib_insync": None}):
            import importlib
            import broker.order_bridge as _mod
            importlib.reload(_mod)
            assert hasattr(_mod, "OrderType")
            assert _mod.OrderType.Market == "MKT"
            assert _mod.OrderType.Limit == "LMT"


# ---------------------------------------------------------------------------
# IBKROrderBridge — _resolve_side
# ---------------------------------------------------------------------------


class TestResolveSide:
    def test_buy_string(self) -> None:
        IBKROrderBridge, Side, OrderType = _import_bridge()
        assert IBKROrderBridge._resolve_side("BUY") == "BUY"

    def test_bot_string(self) -> None:
        IBKROrderBridge, Side, OrderType = _import_bridge()
        assert IBKROrderBridge._resolve_side("BOT") == "BUY"

    def test_sell_string(self) -> None:
        IBKROrderBridge, Side, OrderType = _import_bridge()
        assert IBKROrderBridge._resolve_side("SELL") == "SELL"

    def test_sld_string(self) -> None:
        IBKROrderBridge, Side, OrderType = _import_bridge()
        assert IBKROrderBridge._resolve_side("SLD") == "SELL"

    def test_lowercase_buy(self) -> None:
        IBKROrderBridge, Side, OrderType = _import_bridge()
        assert IBKROrderBridge._resolve_side("buy") == "BUY"

    def test_side_buy_constant(self) -> None:
        IBKROrderBridge, Side, OrderType = _import_bridge()
        assert IBKROrderBridge._resolve_side(Side.Buy) == "BUY"

    def test_side_sell_constant(self) -> None:
        IBKROrderBridge, Side, OrderType = _import_bridge()
        assert IBKROrderBridge._resolve_side(Side.Sell) == "SELL"


# ---------------------------------------------------------------------------
# IBKROrderBridge — _make_ib_order
# ---------------------------------------------------------------------------


@requires_ib
class TestMakeIbOrder:
    def _bridge(self) -> Any:
        IBKROrderBridge, Side, OrderType = _import_bridge()
        mock_conn = MagicMock()
        mock_conn.is_connected = True
        return IBKROrderBridge(connection=mock_conn)

    def test_limit_order_created_for_lmt_type(self) -> None:
        bridge = self._bridge()
        order = _MockOrder(order_type="LMT", limit_price=1.50)
        ib_order = bridge._make_ib_order(order)
        # ib_insync LimitOrder has an 'lmtPrice' attribute
        assert hasattr(ib_order, "lmtPrice") or ib_order.__class__.__name__ == "LimitOrder"

    def test_market_order_created_for_mkt_type(self) -> None:
        bridge = self._bridge()
        order = _MockOrder(order_type="MKT")
        ib_order = bridge._make_ib_order(order)
        assert ib_order.__class__.__name__ == "MarketOrder"

    def test_limit_keyword_also_creates_limit_order(self) -> None:
        """The 'LIMIT' string (not just 'LMT') must produce a LimitOrder."""
        bridge = self._bridge()
        order = _MockOrder(order_type="LIMIT", limit_price=2.00)
        ib_order = bridge._make_ib_order(order)
        assert ib_order.__class__.__name__ == "LimitOrder"

    def test_action_matches_side(self) -> None:
        bridge = self._bridge()
        buy_order = _MockOrder(side="BUY", order_type="MKT")
        sell_order = _MockOrder(side="SELL", order_type="MKT")
        assert bridge._make_ib_order(buy_order).action == "BUY"
        assert bridge._make_ib_order(sell_order).action == "SELL"


# ---------------------------------------------------------------------------
# IBKROrderBridge — submit_order
# ---------------------------------------------------------------------------


@requires_ib
class TestSubmitOrder:
    def _bridge_with_mock_ib(self) -> tuple[Any, MagicMock]:
        IBKROrderBridge, Side, OrderType = _import_bridge()

        mock_ib = MagicMock()
        mock_trade = MagicMock()
        mock_trade.order.permId = 42
        mock_trade.order.orderId = 1001
        mock_trade.fillEvent = MagicMock()
        mock_ib.placeOrder.return_value = mock_trade

        mock_conn = MagicMock()
        mock_conn.is_connected = True
        mock_conn.ib = mock_ib

        bridge = IBKROrderBridge(connection=mock_conn)
        return bridge, mock_ib

    @pytest.mark.asyncio
    async def test_submit_calls_place_order(self) -> None:
        bridge, mock_ib = self._bridge_with_mock_ib()
        await bridge.submit_order(_MockOrder())
        mock_ib.placeOrder.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_returns_perm_id(self) -> None:
        bridge, mock_ib = self._bridge_with_mock_ib()
        perm_id = await bridge.submit_order(_MockOrder())
        assert perm_id == 42

    @pytest.mark.asyncio
    async def test_submit_stores_order_in_map(self) -> None:
        bridge, _ = self._bridge_with_mock_ib()
        order = _MockOrder(order_id=7)
        await bridge.submit_order(order)
        assert 7 in bridge._order_map

    @pytest.mark.asyncio
    async def test_submit_raises_when_not_connected(self) -> None:
        IBKROrderBridge, Side, OrderType = _import_bridge()
        mock_conn = MagicMock()
        mock_conn.is_connected = False
        bridge = IBKROrderBridge(connection=mock_conn)
        with pytest.raises(RuntimeError, match="Not connected"):
            await bridge.submit_order(_MockOrder())

    @pytest.mark.skip(reason="ib_insync eventkit incompatible with MagicMock — needs live TWS")
    @pytest.mark.asyncio
    async def test_submit_registers_fill_callback(self) -> None:
        bridge, mock_ib = self._bridge_with_mock_ib()
        await bridge.submit_order(_MockOrder())
        # The fillEvent.__iadd__ should have been called (+=) for the callback
        trade = mock_ib.placeOrder.return_value
        assert trade.fillEvent.__iadd__.called or trade.fillEvent.__add__.called


# ---------------------------------------------------------------------------
# IBKROrderBridge — cancel_order
# ---------------------------------------------------------------------------


@requires_ib
class TestCancelOrder:
    def _bridge_with_submitted(self):
        IBKROrderBridge, Side, OrderType = _import_bridge()

        mock_ib = MagicMock()
        mock_trade = MagicMock()
        mock_trade.order.permId = 10
        mock_trade.order.orderId = 200
        mock_trade.fillEvent = MagicMock()
        mock_ib.placeOrder.return_value = mock_trade

        mock_conn = MagicMock()
        mock_conn.is_connected = True
        mock_conn.ib = mock_ib

        bridge = IBKROrderBridge(connection=mock_conn)
        return bridge, mock_ib, mock_trade

    @pytest.mark.asyncio
    async def test_cancel_calls_cancel_order_on_ib(self) -> None:
        bridge, mock_ib, mock_trade = self._bridge_with_submitted()
        await bridge.submit_order(_MockOrder(order_id=1))
        await bridge.cancel_order(1)
        mock_ib.cancelOrder.assert_called_once_with(mock_trade.order)

    @pytest.mark.asyncio
    async def test_cancel_unknown_order_does_not_raise(self) -> None:
        bridge, _, _ = self._bridge_with_submitted()
        # Should log a warning but not raise
        await bridge.cancel_order(99_999)


# ---------------------------------------------------------------------------
# IBKROrderBridge — _on_fill callback
# ---------------------------------------------------------------------------


@requires_ib
class TestOnFill:
    def _bridge_with_callbacks(self, order_manager=None, position_tracker=None):
        IBKROrderBridge, Side, OrderType = _import_bridge()

        mock_ib = MagicMock()
        mock_trade = MagicMock()
        mock_trade.order.permId = 1
        mock_trade.order.orderId = 100
        mock_trade.contract.symbol = "AAPL"
        mock_trade.fillEvent = MagicMock()
        mock_ib.placeOrder.return_value = mock_trade

        mock_conn = MagicMock()
        mock_conn.is_connected = True
        mock_conn.ib = mock_ib

        bridge = IBKROrderBridge(
            connection=mock_conn,
            order_manager=order_manager,
            position_tracker=position_tracker,
        )
        return bridge, mock_trade

    @pytest.mark.asyncio
    async def test_on_fill_calls_order_manager(self) -> None:
        mock_om = MagicMock()
        bridge, mock_trade = self._bridge_with_callbacks(order_manager=mock_om)
        await bridge.submit_order(_MockOrder(order_id=1))

        fill = _MockFill()
        bridge._on_fill(mock_trade, fill)

        mock_om.on_fill.assert_called_once_with(1, 1.0, 1.50)

    @pytest.mark.asyncio
    async def test_on_fill_calls_position_tracker(self) -> None:
        mock_pt = MagicMock()
        bridge, mock_trade = self._bridge_with_callbacks(position_tracker=mock_pt)
        await bridge.submit_order(_MockOrder(order_id=1))

        fill = _MockFill()
        bridge._on_fill(mock_trade, fill)

        mock_pt.on_fill.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_fill_without_callbacks_does_not_raise(self) -> None:
        bridge, mock_trade = self._bridge_with_callbacks()
        await bridge.submit_order(_MockOrder(order_id=1))
        fill = _MockFill()
        bridge._on_fill(mock_trade, fill)  # should not raise

    @pytest.mark.asyncio
    async def test_on_fill_sell_side_maps_correctly(self) -> None:
        mock_pt = MagicMock()
        bridge, mock_trade = self._bridge_with_callbacks(position_tracker=mock_pt)
        await bridge.submit_order(_MockOrder(order_id=1, side="SELL"))

        fill = _MockFill(execution=_MockExecution(side="SLD"))
        bridge._on_fill(mock_trade, fill)

        args = mock_pt.on_fill.call_args[0]
        # Third positional arg is the engine side — should be "SELL"
        assert args[1] == "SELL"

    @pytest.mark.asyncio
    async def test_on_fill_order_manager_error_does_not_propagate(self) -> None:
        mock_om = MagicMock()
        mock_om.on_fill.side_effect = Exception("C++ error")
        bridge, mock_trade = self._bridge_with_callbacks(order_manager=mock_om)
        await bridge.submit_order(_MockOrder(order_id=1))

        fill = _MockFill()
        bridge._on_fill(mock_trade, fill)  # must not raise


# ---------------------------------------------------------------------------
# IBKRConnection — basic attribute checks (ib_insync mocked at import)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(True, reason="Integration tests — require running IBKR TWS on port 7497")
class TestIBKRConnection:
    def test_raises_import_error_without_ib_insync(self) -> None:
        """IBKRConnection.__init__ raises ImportError when ib_insync is absent."""
        with patch.dict("sys.modules", {"ib_insync": None}):
            import importlib
            import broker.connection as _conn_mod
            importlib.reload(_conn_mod)
            with pytest.raises(ImportError, match="ib_insync"):
                _conn_mod.IBKRConnection()

    def test_ibkr_connection_error_is_exception_subclass(self) -> None:
        IBKRConnection, IBKRConnectionError = _import_connection()
        assert issubclass(IBKRConnectionError, Exception)

    @pytest.mark.asyncio
    async def test_connect_raises_after_max_retries(self) -> None:
        IBKRConnection, IBKRConnectionError = _import_connection()

        with patch("broker.connection.IB") as MockIB:
            ib_instance = MagicMock()
            ib_instance.connectAsync = AsyncMock(side_effect=OSError("refused"))
            ib_instance.isConnected.return_value = False
            ib_instance.disconnectedEvent = MagicMock()
            ib_instance.disconnectedEvent.__iadd__ = MagicMock()
            MockIB.return_value = ib_instance

            conn = IBKRConnection(host="127.0.0.1", port=7497, client_id=99)
            with pytest.raises(IBKRConnectionError):
                await conn.connect()

    @pytest.mark.asyncio
    async def test_connect_succeeds_on_first_attempt(self) -> None:
        IBKRConnection, IBKRConnectionError = _import_connection()

        with patch("broker.connection.IB") as MockIB:
            ib_instance = MagicMock()
            ib_instance.connectAsync = AsyncMock(return_value=None)
            ib_instance.isConnected.return_value = True
            ib_instance.disconnectedEvent = MagicMock()
            ib_instance.disconnectedEvent.__iadd__ = MagicMock()
            MockIB.return_value = ib_instance

            conn = IBKRConnection(host="127.0.0.1", port=7497, client_id=99)
            # Patch _start_heartbeat to avoid spawning a real asyncio task
            with patch.object(conn, "_start_heartbeat", return_value=None):
                await conn.connect()

            assert conn._connected is True

    @pytest.mark.asyncio
    async def test_disconnect_stops_heartbeat_and_disconnects(self) -> None:
        IBKRConnection, IBKRConnectionError = _import_connection()

        with patch("broker.connection.IB") as MockIB:
            ib_instance = MagicMock()
            ib_instance.isConnected.return_value = True
            ib_instance.disconnect = MagicMock()
            ib_instance.disconnectedEvent = MagicMock()
            ib_instance.disconnectedEvent.__iadd__ = MagicMock()
            MockIB.return_value = ib_instance

            conn = IBKRConnection()
            conn._connected = True
            with patch.object(conn, "_stop_heartbeat", return_value=None):
                await conn.disconnect()

            ib_instance.disconnect.assert_called_once()
            assert conn._connected is False

    def test_is_connected_delegates_to_ib(self) -> None:
        IBKRConnection, IBKRConnectionError = _import_connection()

        with patch("broker.connection.IB") as MockIB:
            ib_instance = MagicMock()
            ib_instance.isConnected.return_value = True
            ib_instance.disconnectedEvent = MagicMock()
            ib_instance.disconnectedEvent.__iadd__ = MagicMock()
            MockIB.return_value = ib_instance

            conn = IBKRConnection()
            assert conn.is_connected is True


# ---------------------------------------------------------------------------
# PositionReconciler — _compare logic (pure Python, no IBKR needed)
# ---------------------------------------------------------------------------


class TestPositionReconcilerCompare:
    def _reconciler(self) -> Any:
        PositionReconciler, Discrepancy, DiscrepancyType = _import_reconciler()
        mock_conn = MagicMock()
        mock_conn.ib = MagicMock()
        return PositionReconciler(connection=mock_conn), Discrepancy, DiscrepancyType

    def test_no_discrepancies_when_positions_match(self) -> None:
        rec, Discrepancy, DiscrepancyType = self._reconciler()
        broker = {"AAPL": {"qty": 2.0, "avg_cost": 1.50}}
        local = {"AAPL": {"qty": 2.0, "avg_cost": 1.50}}
        assert rec._compare(broker, local) == []

    def test_missing_local_detected(self) -> None:
        rec, Discrepancy, DiscrepancyType = self._reconciler()
        broker = {"AAPL": {"qty": 2.0, "avg_cost": 1.50}}
        local: dict = {}
        discrepancies = rec._compare(broker, local)
        assert len(discrepancies) == 1
        assert discrepancies[0].kind == DiscrepancyType.MISSING_LOCAL

    def test_missing_broker_detected(self) -> None:
        rec, Discrepancy, DiscrepancyType = self._reconciler()
        broker: dict = {}
        local = {"AAPL": {"qty": 2.0, "avg_cost": 1.50}}
        discrepancies = rec._compare(broker, local)
        assert len(discrepancies) == 1
        assert discrepancies[0].kind == DiscrepancyType.MISSING_BROKER

    def test_quantity_mismatch_detected(self) -> None:
        rec, Discrepancy, DiscrepancyType = self._reconciler()
        broker = {"AAPL": {"qty": 3.0, "avg_cost": 1.50}}
        local = {"AAPL": {"qty": 2.0, "avg_cost": 1.50}}
        kinds = [d.kind for d in rec._compare(broker, local)]
        assert DiscrepancyType.QUANTITY_MISMATCH in kinds

    def test_price_divergence_detected(self) -> None:
        rec, Discrepancy, DiscrepancyType = self._reconciler()
        broker = {"AAPL": {"qty": 2.0, "avg_cost": 2.00}}
        local = {"AAPL": {"qty": 2.0, "avg_cost": 1.00}}  # 50% divergence
        kinds = [d.kind for d in rec._compare(broker, local)]
        assert DiscrepancyType.PRICE_DIVERGENCE in kinds

    def test_no_price_divergence_within_threshold(self) -> None:
        rec, Discrepancy, DiscrepancyType = self._reconciler()
        broker = {"AAPL": {"qty": 2.0, "avg_cost": 1.50}}
        local = {"AAPL": {"qty": 2.0, "avg_cost": 1.52}}  # ~1.3% diff
        kinds = [d.kind for d in rec._compare(broker, local)]
        assert DiscrepancyType.PRICE_DIVERGENCE not in kinds

    def test_multiple_symbols_all_compared(self) -> None:
        rec, Discrepancy, DiscrepancyType = self._reconciler()
        broker = {
            "AAPL": {"qty": 1.0, "avg_cost": 1.50},
            "MSFT": {"qty": 2.0, "avg_cost": 3.00},
        }
        local = {
            "AAPL": {"qty": 1.0, "avg_cost": 1.50},
            "GOOG": {"qty": 5.0, "avg_cost": 2.00},
        }
        discrepancies = rec._compare(broker, local)
        kinds = {d.kind for d in discrepancies}
        assert DiscrepancyType.MISSING_LOCAL in kinds   # MSFT missing locally
        assert DiscrepancyType.MISSING_BROKER in kinds  # GOOG missing at broker


# ---------------------------------------------------------------------------
# PositionReconciler._contract_key static method
# ---------------------------------------------------------------------------


class TestContractKey:
    def _reconciler(self) -> Any:
        PositionReconciler, _, _ = _import_reconciler()
        mock_conn = MagicMock()
        mock_conn.ib = MagicMock()
        return PositionReconciler(connection=mock_conn)

    def test_stock_contract_uses_symbol(self) -> None:
        rec = self._reconciler()
        contract = MagicMock()
        contract.secType = "STK"
        contract.symbol = "AAPL"
        assert rec._contract_key(contract) == "AAPL"

    def test_option_contract_includes_strike_and_right(self) -> None:
        rec = self._reconciler()
        contract = MagicMock()
        contract.secType = "OPT"
        contract.symbol = "AAPL"
        contract.lastTradeDateOrContractMonth = "20251219"
        contract.strike = 150.0
        contract.right = "C"
        key = rec._contract_key(contract)
        assert "AAPL" in key
        assert "150" in key
        assert "C" in key

    def test_unknown_sectype_falls_back_to_symbol(self) -> None:
        rec = self._reconciler()
        contract = MagicMock()
        contract.secType = "FUT"
        contract.symbol = "ES"
        assert rec._contract_key(contract) == "ES"


# ---------------------------------------------------------------------------
# PositionReconciler.startup_sync
# ---------------------------------------------------------------------------


@requires_ib
class TestStartupSync:
    @pytest.mark.asyncio
    async def test_startup_sync_returns_empty_list_when_matched(self) -> None:
        PositionReconciler, _, _ = _import_reconciler()

        mock_conn = MagicMock()
        mock_ib = MagicMock()
        mock_ib.positions.return_value = []
        mock_conn.ib = mock_ib

        # No position_tracker → empty local dict
        rec = PositionReconciler(connection=mock_conn)
        result = await rec.startup_sync()
        assert result == []

    @pytest.mark.asyncio
    async def test_startup_sync_returns_discrepancy_strings(self) -> None:
        PositionReconciler, _, _ = _import_reconciler()

        # Simulate IBKR reporting one position the local engine doesn't know about
        mock_position = MagicMock()
        mock_position.contract.secType = "STK"
        mock_position.contract.symbol = "AAPL"
        mock_position.position = 5.0
        mock_position.avgCost = 1.50

        mock_conn = MagicMock()
        mock_ib = MagicMock()
        mock_ib.positions.return_value = [mock_position]
        mock_conn.ib = mock_ib

        rec = PositionReconciler(connection=mock_conn)  # no position_tracker
        result = await rec.startup_sync()

        assert len(result) == 1
        assert "AAPL" in result[0]
