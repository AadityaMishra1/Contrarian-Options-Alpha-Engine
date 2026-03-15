"""Position reconciliation between the C++ PositionTracker and IBKR.

Fetches live positions from IBKR on startup and on a configurable interval,
compares them to what the local engine believes it holds, and logs any
discrepancies.  Discrepancies are categorised but never auto-corrected —
human review is required for position mismatches.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from .connection import IBKRConnection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Discrepancy types
# ---------------------------------------------------------------------------

class DiscrepancyType(Enum):
    """Classification of position discrepancies found during reconciliation."""

    MISSING_LOCAL = auto()    # IBKR has a position the engine does not know about
    MISSING_BROKER = auto()   # Engine holds a position IBKR does not report
    QUANTITY_MISMATCH = auto()  # Both sides have the position but sizes differ
    PRICE_DIVERGENCE = auto()  # Average cost differs materially (>5%)


@dataclass
class Discrepancy:
    """A single position discrepancy record.

    Attributes:
        kind: The category of mismatch.
        symbol: Underlying or full contract description.
        local_qty: Quantity in the C++ PositionTracker (None if absent).
        broker_qty: Quantity reported by IBKR (None if absent).
        local_avg_cost: Average cost in the engine (None if absent).
        broker_avg_cost: Average cost reported by IBKR (None if absent).
        description: Human-readable summary of the discrepancy.
    """

    kind: DiscrepancyType
    symbol: str
    local_qty: float | None = None
    broker_qty: float | None = None
    local_avg_cost: float | None = None
    broker_avg_cost: float | None = None
    description: str = ""

    def __str__(self) -> str:
        return (
            f"[{self.kind.name}] {self.symbol}: "
            f"local_qty={self.local_qty}, broker_qty={self.broker_qty} — "
            f"{self.description}"
        )


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------

class PositionReconciler:
    """Syncs the C++ PositionTracker with IBKR's live position report.

    Args:
        connection: An active :class:`IBKRConnection` instance.
        position_tracker: C++ PositionTracker pybind11 object, or None.
                          If None, reconciliation compares against an empty
                          local set (all IBKR positions are flagged as
                          MISSING_LOCAL).
        check_interval: Seconds between periodic reconciliation checks.
                        Defaults to 300 (5 minutes).
    """

    _PRICE_DIVERGENCE_THRESHOLD: float = 0.05  # 5 % average-cost divergence

    def __init__(
        self,
        connection: IBKRConnection,
        position_tracker: Any | None = None,
        check_interval: int = 300,
    ) -> None:
        self._conn = connection
        self._position_tracker = position_tracker
        self._check_interval = check_interval
        self._task: asyncio.Task | None = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def startup_sync(self) -> list[str]:
        """Perform a full reconciliation and return human-readable discrepancy strings.

        Steps:
            1. Fetch all positions from IBKR via ``ib.positions()``.
            2. Fetch all positions from the local PositionTracker (if set).
            3. Compare and categorise any mismatches.
            4. Log each discrepancy at WARNING level.

        Returns:
            A list of discrepancy description strings (empty if all match).
        """
        logger.info("Starting position reconciliation.")

        broker_positions = self._fetch_broker_positions()
        local_positions = self._fetch_local_positions()

        discrepancies = self._compare(broker_positions, local_positions)

        if not discrepancies:
            logger.info("Reconciliation complete: no discrepancies found.")
        else:
            for d in discrepancies:
                logger.warning("Position discrepancy: %s", d)

        return [str(d) for d in discrepancies]

    async def periodic_check(self) -> None:
        """Continuously run reconciliation every ``check_interval`` seconds."""
        while True:
            await asyncio.sleep(self._check_interval)
            try:
                await self.startup_sync()
            except Exception as exc:
                logger.error("Periodic reconciliation error: %s", exc)

    async def start(self) -> None:
        """Start the background periodic reconciliation task."""
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self.periodic_check())
            logger.info(
                "Position reconciliation task started (interval=%ds).",
                self._check_interval,
            )

    async def stop(self) -> None:
        """Cancel the background reconciliation task."""
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            logger.info("Position reconciliation task stopped.")
        self._task = None

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _fetch_broker_positions(self) -> dict[str, dict]:
        """Return IBKR positions keyed by a canonical symbol string.

        Returns:
            Mapping of symbol -> {"qty": float, "avg_cost": float}.
        """
        ib = self._conn.ib
        result: dict[str, dict] = {}
        try:
            positions = ib.positions()
            for pos in positions:
                contract = pos.contract
                # Build a canonical key: e.g. "AAPL_20250117_150_P" for an option
                key = self._contract_key(contract)
                result[key] = {
                    "qty": float(pos.position),
                    "avg_cost": float(pos.avgCost),
                }
        except Exception as exc:
            logger.error("Failed to fetch IBKR positions: %s", exc)
        return result

    def _fetch_local_positions(self) -> dict[str, dict]:
        """Return positions from the C++ PositionTracker.

        Falls back to an empty dict when no tracker is configured.

        Returns:
            Mapping of symbol -> {"qty": float, "avg_cost": float}.
        """
        if self._position_tracker is None:
            return {}
        result: dict[str, dict] = {}
        try:
            all_pos = self._position_tracker.get_all_positions()
            for pos in all_pos:
                key = getattr(pos, "symbol", str(pos))
                result[key] = {
                    "qty": float(getattr(pos, "quantity", 0.0)),
                    "avg_cost": float(getattr(pos, "avg_cost", 0.0)),
                }
        except Exception as exc:
            logger.error("Failed to fetch local positions: %s", exc)
        return result

    # ------------------------------------------------------------------
    # Comparison logic
    # ------------------------------------------------------------------

    def _compare(
        self,
        broker: dict[str, dict],
        local: dict[str, dict],
    ) -> list[Discrepancy]:
        """Produce a list of Discrepancy objects from two position maps.

        Args:
            broker: Positions reported by IBKR.
            local: Positions held by the local engine.

        Returns:
            All detected discrepancies.
        """
        discrepancies: list[Discrepancy] = []
        all_symbols = set(broker) | set(local)

        for symbol in all_symbols:
            in_broker = symbol in broker
            in_local = symbol in local

            if in_broker and not in_local:
                discrepancies.append(
                    Discrepancy(
                        kind=DiscrepancyType.MISSING_LOCAL,
                        symbol=symbol,
                        broker_qty=broker[symbol]["qty"],
                        broker_avg_cost=broker[symbol]["avg_cost"],
                        description="IBKR holds position not tracked locally.",
                    )
                )
                continue

            if in_local and not in_broker:
                discrepancies.append(
                    Discrepancy(
                        kind=DiscrepancyType.MISSING_BROKER,
                        symbol=symbol,
                        local_qty=local[symbol]["qty"],
                        local_avg_cost=local[symbol]["avg_cost"],
                        description="Engine tracks position not found at IBKR.",
                    )
                )
                continue

            # Both sides have the symbol — check quantity
            b_qty = broker[symbol]["qty"]
            l_qty = local[symbol]["qty"]
            if abs(b_qty - l_qty) > 1e-6:
                discrepancies.append(
                    Discrepancy(
                        kind=DiscrepancyType.QUANTITY_MISMATCH,
                        symbol=symbol,
                        local_qty=l_qty,
                        broker_qty=b_qty,
                        description=f"Quantity mismatch: local={l_qty}, broker={b_qty}.",
                    )
                )

            # Check average cost divergence
            b_cost = broker[symbol]["avg_cost"]
            l_cost = local[symbol]["avg_cost"]
            if b_cost > 0 and abs(b_cost - l_cost) / b_cost > self._PRICE_DIVERGENCE_THRESHOLD:
                discrepancies.append(
                    Discrepancy(
                        kind=DiscrepancyType.PRICE_DIVERGENCE,
                        symbol=symbol,
                        local_qty=l_qty,
                        broker_qty=b_qty,
                        local_avg_cost=l_cost,
                        broker_avg_cost=b_cost,
                        description=(
                            f"Average cost divergence: local={l_cost:.4f}, broker={b_cost:.4f}."
                        ),
                    )
                )

        return discrepancies

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _contract_key(contract: Any) -> str:
        """Build a canonical string key for an ib_insync Contract.

        Args:
            contract: An ib_insync Contract (Stock, Option, etc.).

        Returns:
            A human-readable, unique key for the contract.
        """
        sec_type: str = getattr(contract, "secType", "")
        symbol: str = getattr(contract, "symbol", "UNKNOWN")

        if sec_type == "OPT":
            expiry = getattr(contract, "lastTradeDateOrContractMonth", "")
            strike = getattr(contract, "strike", 0.0)
            right = getattr(contract, "right", "")
            return f"{symbol}_{expiry}_{strike}_{right}"

        return symbol
