"""Bridges C++ Order structs (via pybind11) to ib_insync orders.

Translates the engine's internal Side / OrderType enums into ib_insync
contract and order objects, submits them through an IBKRConnection, and
routes fill callbacks back to the C++ OrderManager / PositionTracker when
they are available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

try:
    from ib_insync import IB, LimitOrder, MarketOrder, Trade  # noqa: F401
    from ib_insync import Option as IBOption

    HAS_IB_INSYNC = True
except ImportError:
    HAS_IB_INSYNC = False
    IB = None  # type: ignore[assignment,misc]
    Trade = None  # type: ignore[assignment,misc]

from .connection import IBKRConnection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight enum stand-ins used when the C++ engine is absent
# ---------------------------------------------------------------------------

class Side:
    """Order side constants matching the C++ engine enum values."""

    Buy = "BUY"
    Sell = "SELL"


class OrderType:
    """Order type constants matching the C++ engine enum values."""

    Market = "MKT"
    Limit = "LMT"


# ---------------------------------------------------------------------------
# Internal tracking record
# ---------------------------------------------------------------------------

@dataclass
class _OrderRecord:
    """Internal mapping between a C++ order and an IBKR Trade object.

    Attributes:
        cpp_order_id: Identifier from the C++ OrderManager (or caller).
        ibkr_trade: The ib_insync Trade returned by ib.placeOrder().
        symbol: Underlying ticker symbol.
        side: "BUY" or "SELL".
    """

    cpp_order_id: int
    ibkr_trade: Any  # ib_insync.Trade
    symbol: str
    side: str


# ---------------------------------------------------------------------------
# Main bridge class
# ---------------------------------------------------------------------------

class IBKROrderBridge:
    """Converts engine Order objects to ib_insync orders and handles fills.

    Args:
        connection: An active :class:`IBKRConnection` instance.
        order_manager: C++ OrderManager pybind11 object, or None.
        position_tracker: C++ PositionTracker pybind11 object, or None.

    When ``order_manager`` or ``position_tracker`` are None the bridge
    still works: fills are logged but callbacks are skipped.
    """

    def __init__(
        self,
        connection: IBKRConnection,
        order_manager: Any | None = None,
        position_tracker: Any | None = None,
    ) -> None:
        if not HAS_IB_INSYNC:
            raise ImportError(
                "ib_insync is not installed. Run: pip install ib_insync"
            )

        self._conn = connection
        self._order_manager = order_manager
        self._position_tracker = position_tracker

        # cpp_order_id -> _OrderRecord
        self._order_map: dict[int, _OrderRecord] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit_order(self, order: Any) -> int:
        """Submit an order to IBKR and return the IBKR permId.

        Args:
            order: A C++ Order object (or duck-typed equivalent) with
                   attributes: ``order_id``, ``symbol``, ``expiry``,
                   ``strike``, ``right`` ("C"/"P"), ``side`` (Side.*),
                   ``order_type`` (OrderType.*), and optionally
                   ``limit_price``, ``quantity``.

        Returns:
            The IBKR permanent order ID (``trade.order.permId``).

        Raises:
            RuntimeError: If the connection is not active.
        """
        if not self._conn.is_connected:
            raise RuntimeError("Not connected to IBKR. Call connect() first.")

        contract = self._make_contract(order)
        ib_order = self._make_ib_order(order)

        ib = self._conn.ib
        trade: Trade = ib.placeOrder(contract, ib_order)

        # Register fill callback on this specific Trade
        trade.fillEvent += self._on_fill

        record = _OrderRecord(
            cpp_order_id=order.order_id,
            ibkr_trade=trade,
            symbol=getattr(order, "symbol", ""),
            side=self._resolve_side(order.side),
        )
        self._order_map[order.order_id] = record

        logger.info(
            "Submitted order: cpp_id=%d symbol=%s side=%s qty=%s ibkr_id=%s",
            order.order_id,
            getattr(order, "symbol", ""),
            record.side,
            getattr(order, "quantity", "?"),
            trade.order.orderId,
        )
        return trade.order.permId

    async def cancel_order(self, order_id: int) -> None:
        """Cancel an open order by its C++ order ID.

        Args:
            order_id: The identifier originally passed in the Order object.

        Raises:
            KeyError: If no such order is tracked.
        """
        record = self._order_map.get(order_id)
        if record is None:
            logger.warning("cancel_order: unknown order_id %d", order_id)
            return

        ib = self._conn.ib
        ib.cancelOrder(record.ibkr_trade.order)
        logger.info("Cancelled IBKR order for cpp_id=%d", order_id)

    # ------------------------------------------------------------------
    # Fill callback
    # ------------------------------------------------------------------

    def _on_fill(self, trade: Any, fill: Any) -> None:
        """Handle a fill event from ib_insync.

        Looks up the C++ order ID from the trade, then notifies
        ``order_manager`` and ``position_tracker`` if they are set.

        Args:
            trade: The ib_insync Trade object that generated the fill.
            fill: The ib_insync Fill (contains Execution and CommissionReport).
        """
        # Reverse-lookup the cpp order ID via the IBKR order ID
        cpp_order_id: int | None = None
        for oid, record in self._order_map.items():
            if record.ibkr_trade is trade:
                cpp_order_id = oid
                break

        exec_ = fill.execution
        filled_qty: float = exec_.shares
        filled_price: float = exec_.avgPrice
        symbol: str = trade.contract.symbol if trade.contract else ""
        side: str = exec_.side  # "BOT" or "SLD" from IBKR

        logger.info(
            "Fill received: cpp_id=%s symbol=%s side=%s qty=%.0f price=%.4f",
            cpp_order_id,
            symbol,
            side,
            filled_qty,
            filled_price,
        )

        if cpp_order_id is not None and self._order_manager is not None:
            try:
                self._order_manager.on_fill(cpp_order_id, filled_qty, filled_price)
            except Exception as exc:
                logger.error("order_manager.on_fill error: %s", exc)

        if self._position_tracker is not None:
            try:
                # Normalise IBKR side strings to engine convention
                engine_side = Side.Buy if side.upper() in ("BOT", "BUY") else Side.Sell
                self._position_tracker.on_fill(symbol, engine_side, filled_qty, filled_price)
            except Exception as exc:
                logger.error("position_tracker.on_fill error: %s", exc)

    # ------------------------------------------------------------------
    # Contract / order builders
    # ------------------------------------------------------------------

    def _make_contract(self, order: Any) -> Any:
        """Build an ib_insync Option contract from an engine Order.

        Args:
            order: Engine order with ``symbol``, ``expiry`` (YYYYMMDD str),
                   ``strike`` (float), and ``right`` ("C" or "P").

        Returns:
            An ib_insync ``Option`` contract with exchange="SMART".
        """
        symbol: str = getattr(order, "symbol", "")
        expiry: str = getattr(order, "expiry", "")
        strike: float = float(getattr(order, "strike", 0.0))
        right: str = getattr(order, "right", "P")  # "C" or "P"

        contract = IBOption(
            symbol,
            expiry,
            strike,
            right,
            exchange="SMART",
            currency="USD",
        )
        return contract

    def _make_ib_order(self, order: Any) -> Any:
        """Build an ib_insync order object from an engine Order.

        Args:
            order: Engine order with ``side``, ``order_type``, ``quantity``,
                   and optionally ``limit_price``.

        Returns:
            An ib_insync ``MarketOrder`` or ``LimitOrder``.
        """
        action: str = self._resolve_side(order.side)
        qty: float = float(getattr(order, "quantity", 1))
        order_type = getattr(order, "order_type", OrderType.Market)

        # Normalise order_type — handles both enum-like objects and string constants
        ot_str: str = (
            order_type if isinstance(order_type, str) else str(order_type)
        ).upper()

        if "LMT" in ot_str or "LIMIT" in ot_str:
            limit_price: float = float(getattr(order, "limit_price", 0.0))
            ib_order = LimitOrder(action, qty, limit_price)
        else:
            ib_order = MarketOrder(action, qty)

        return ib_order

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_side(side: Any) -> str:
        """Resolve a C++ Side enum or string to "BUY" / "SELL".

        Args:
            side: Either a C++ enum value, the Python Side class attribute,
                  or a plain string.

        Returns:
            "BUY" or "SELL".
        """
        if isinstance(side, str):
            upper = side.upper()
            if upper in ("BUY", "BOT"):
                return "BUY"
            return "SELL"

        # Handle C++ enum: compare to known sentinel values
        side_str = str(side).upper()
        if "BUY" in side_str or "BOT" in side_str:
            return "BUY"
        return "SELL"
